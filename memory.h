#ifndef __CNN_HLS_MEMORY_H__
#define __CNN_HLS_MEMORY_H__
#include "types.h"
#include "utils.h"
#include <hlslib/xilinx/TreeReduce.h>
#include <hlslib/xilinx/Operators.h>
#include "operators.h"

/*
 * ddr3 512MB ram (256Mx16) (MT41K512M16HA-125:A) (MT41K256M16TW-107:P ??)
 * 8 banks * 64K (32K?) rows * 1K columns * 16 bits
 * ref: https://d2m32eurp10079.cloudfront.net/Download/pynqz2_user_manual_v1_0.pdf
 * 	    https://media-www.micron.com/-/media/client/global/documents/products/data-sheet/dram/ddr3/4gb_ddr3l.pdf?rev=8d4b345161424b60bbe4886434cbccf4
 *
 * Page size: 2KB
 * Minimum burst size: 16B (8 (ddr3 prefetch depth) * 16 bits)
 * cache line size should be: 16B <= size <= 2KB
 */

template<typename T, uint_t ADDR_SIZE>
class MainMemory_ref {
	/*
	 * reference class implementation for MainMemory
	 * do not use
	 */
public:
	using RandomAccessType = void&; // memory type
	using addr_t = ap_uint<ADDR_SIZE>;
private:
	RandomAccessType ram;
public:
	struct ref_struct {
		RandomAccessType ram;
		const addr_t addr;
		operator T() const {
#pragma HLS INLINE
			return ram[addr]; // ram.get(addr), or ram.get(base + addr * sizeof(T)), ...
		}
		ref_struct operator=(T value) const {
#pragma HLS INLINE
			ram[addr] = value; //ram.set(addr, value), ...
			return *this;
		}
	};
	ref_struct operator[](addr_t addr) const {
#pragma HLS INLINE
		return {this->ram, addr};
	}
};

template<typename T, typename MainMemory,
		uint_t ADDR_SIZE, uint_t OFFSET_BITS, uint_t LINE_BITS,
		uint_t NUM_WAYS, uint_t ASSIGN_BITS, uint_t TIME_BITS>
class ram_cache {
public:
	static const uint_t TAG_BITS = ADDR_SIZE - OFFSET_BITS - LINE_BITS;
	static const uint_t NUM_VALUES = 1 << OFFSET_BITS;
	static const uint_t NUM_LINES = 1 << LINE_BITS;
	static const uint_t BUS_SIZE = sizeof(T);
	static const uint_t LINE_SIZE = NUM_VALUES * BUS_SIZE;
	static const uint_t WAY_SIZE = NUM_LINES * LINE_SIZE;
	static const uint_t CACHE_SIZE = NUM_WAYS * WAY_SIZE;

	typedef ram_cache this_class;

	using addr_t = ap_uint<ADDR_SIZE>;
	using offset_t = ap_uint<OFFSET_BITS>;
	using line_t = ap_uint<LINE_BITS>;
	using tag_t = ap_uint<TAG_BITS>;
	using assign_t = ap_uint<ASSIGN_BITS>;
	using way_t = ap_uint<log2_ceil<uint_t>(NUM_WAYS)>;
	using time_t = ap_uint<TIME_BITS>;

	static offset_t offset_part(addr_t addr) {
#pragma HLS INLINE
		return addr(OFFSET_BITS - 1, 0);
	}
	static line_t line_part(addr_t addr) {
#pragma HLS INLINE
		return addr(OFFSET_BITS + LINE_BITS - 1, OFFSET_BITS);
	}
	static tag_t tag_part(addr_t addr) {
#pragma HLS INLINE
		return addr(ADDR_SIZE - 1, OFFSET_BITS + LINE_BITS);
	}
	static addr_t make_addr(tag_t tag, line_t line, offset_t offset) {
#pragma HLS INLINE
		return (tag, (line, offset).get());
	}

protected:
	struct line_struct {
		T buf[NUM_VALUES];
		bool valid = false, dirty = false;
		tag_t tag;
		time_t last_use;
	};
	struct way_struct {
		line_struct buf[NUM_LINES];
		assign_t assign;
	};


private:
	way_struct buf[NUM_WAYS];
#pragma HLS ARRAY_PARTITION variable=buf complete
	MainMemory main_memory;

	time_t current_time[NUM_LINES];

protected:
	bool find_tag(line_t line, tag_t tag, way_t &way) const {
		using op_t = MaxOperand<bool, way_t>;
		struct match_ram {
			const way_struct (&buf)[NUM_WAYS];
			line_t line;
			tag_t tag;
			op_t operator[](way_t way) const {
#pragma HLS INLINE
				auto &line_ref = buf[way].buf[line];
				return {line_ref.valid && line_ref.tag == tag, way};
			}
		} ram{buf, line, tag};
		op_t ret = hlslib::TreeReduce<op_t, hlslib::op::Max<op_t>, NUM_WAYS>(ram);
		way = ret.value;
		return ret.key;
	}
	way_t find_replacement(line_t line, assign_t referer) const {
		using op_t = MaxOperand<time_t, way_t>;
		struct policy_ram {
			const way_struct (&buf)[NUM_WAYS];
			line_t line;
			assign_t referer;
			op_t operator[](way_t way) const {
#pragma HLS INLINE
				auto &line_ref = buf[way].buf[line];
				return {
					line_ref.assign == referer ?
							line_ref.valid ?
									current_time[line] - line_ref.last_use : // normal
									-1 : // assigned and invalid
							line_ref.valid ?
									0 : // not assigned and invalid, prohibited
									-2, // not assigned and invalid, prefer assigned
					way
				};
			}
		} ram{buf, line, referer};
		op_t ret = hlslib::TreeReduce<op_t, hlslib::op::Max<op_t>, NUM_WAYS>(ram);
		return ret.value;
	}
	bool try_get(addr_t addr, T &value) {
		auto offset = offset_part(addr);
		auto line = line_part(addr);
		auto tag = tag_part(addr);
		way_t way;
		if (find_tag(line, tag, way)) {
			auto &line_ref = buf[way].buf[line];
			value = line_ref.buf[offset];
			line_ref.last_use = current_time++;
			return true;
		}
		return false;
	}
	bool try_set(addr_t addr, T &&value) {
		auto offset = offset_part(addr);
		auto line = line_part(addr);
		auto tag = tag_part(addr);
		way_t way;
		if (find_tag(line, tag, way)) {
			auto &line_ref = buf[way].buf[line];
			line_ref.buf[offset] = value;
			line_ref.dirty = true;
			line_ref.last_use = current_time++;
			return true;
		}
		return false;
	}
	void allocate(line_t line, tag_t tag, assign_t referer) {
		way_t way;

		//replacement policy
		way = find_replacement(line, referer);

		line_struct &line_ref = buf[way].buf[line];

		//write-back
		if (line_ref.dirty) {
			offset_t offset = 0;
			auto addr = ((line_ref.tag, line).get(), offset);
			do {
				main_memory[addr] = line_ref.buf[offset];
			} while (++offset);
		}

		line_ref.tag = tag;

		//copy
		{
			offset_t offset = 0;
			auto addr = ((tag, line).get(), offset);
			do {
				line_ref.buf[offset] = main_memory[addr];
			} while (++offset);
		}
	}

public:
	ram_cache() {
	}

	T get(addr_t addr, assign_t referer) {
		T ret;
		while (!this->try_get(addr, ret)) {
			this->allocate(line_part(addr), tag_part(addr), referer);
		}
		return ret;
	}
	void set(addr_t addr, T &&value, assign_t referer) {
		while (!this->try_set(addr, value)) {
			this->allocate(line_part(addr), tag_part(addr), referer);
		}
	}

	struct ref_struct {
		this_class &cache;
		const assign_t referer;
		const addr_t addr;
		operator T() const {
#pragma HLS INLINE
			return cache.get(addr, referer);
		}
		ref_struct operator=(T value) const {
#pragma HLS INLINE
			cache.set(addr, value, referer);
			return *this;
		}
	};

	class referer_bound {
	private:
		referer_bound(this_class &p_cache, assign_t p_referer)
				: cache(p_cache), referer(p_referer) {}
	public:
		ref_struct operator[](addr_t addr) const {
#pragma HLS INLINE
			return {cache, referer, addr};
		}
	private:
		this_class &cache;
		const assign_t referer;
	};

	referer_bound bind(assign_t referer) {
		return referer_bound(*this, referer);
	}

	void configure(const assign_t assign_vector[NUM_WAYS]) {
		for (way_t i = 0; i < NUM_WAYS; i++) {
#pragma HLS UNROLL
			buf[i].assign = assign_vector[i];
		}
	}

	ram_cache(this_class const &) = delete;
	ram_cache(this_class &&) = delete;
	this_class &operator=(this_class const &) = delete;
	this_class &operator=(this_class &&) = delete;

};
#endif

