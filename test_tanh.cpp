#include <iostream>
#include <cstring>
#include <algorithm>
#include <math.h>

#include "types.h"
#include "cnn_functions.h"

minibatch_t a[100000];
minibatch_t b[100000];
minibatch_t ori[100000];
value_t merror,my_max_tanh;
int_t num;

value_t my_abs(value_t x)
{
	return x>=(value_t)0?x:(value_t)(-x);
}

int main() {
	//*
	for(int_t i=-40000;i<=40000;i++)
	{
		a[i+40000][0]=i/1000.0;
		ori[i+40000][0]=tanh(i/1000.0);
	}
	cnn_Tanh<0, 1, value_t>(a, b, 80001);
	for(int_t i=0;i<=80000;i++)
	{
		if(i==40000) continue;
		value_t now=my_abs(value_t(ori[i][0])-value_t(b[i][0]))/my_abs(value_t(ori[i][0]));
		std::cout<<(i-40000)/1000.0<<' '<<double(value_t(b[i][0]))<<' '<<double(value_t(ori[i][0]))<<std::endl;
		//if(merror<now) merror=now,num=i,my_max_tanh=value_t(b[i][0]);
	}
	//*/

	/*value_t p;

	for(int_t i=-170000;i<=170000;i++)
	{
		p=my_exp((value_t)(i/1000.0));
		std::cout<<i/1000.0<<' '<<double(p)<<' '<<exp(i/1000.0)<<std::endl;
	}*/
	//std::cout<<"Error : "<<(double)merror<<' '<<num<<' '<<my_max_tanh<<' '<<tanh((num-40000)/1000.0)<<std::endl;
	return 0;
}
