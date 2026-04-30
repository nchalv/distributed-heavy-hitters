#ifndef _cmsketch_H
#define _cmsketch_H
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <string>
#include <cstring>
#include <unordered_map>
#include "BaseSketch.h"
#include "BOBHASH32.h"
#include "params.h"
#include "ssummary.h"
#include "BOBHASH64.h"
#define CM_d 4
#define rep(i,a,n) for(int i=a;i<=n;i++)
using namespace std;
//use CM sketch+space summary to make the result invertible.
class cmsketch : public sketch::BaseSketch{
public:
	ssummary *ss;
	struct node { int C; } HK[CM_d][MAX_MEM+10];
	BOBHash64 * bobhash;
	int  M2, K, d;

	cmsketch(int Mem,int Knum) :M2(M2){
		M2=Mem;
		K=Knum;
		ss = new ssummary(K); ss->clear(); 
		bobhash = new BOBHash64(1005); 
	}
	
	void clear()
	{
		for (int i = 0; i < CM_d; i++)
			for (int j = 0; j <= M2 + 5; j++)
				HK[i][j].C = 0;
	}

	unsigned long long Hash(string ST)
	{
		return (bobhash->run(ST.c_str(), ST.size()));
	}
	
	void Insert(const string x)
	{
		//insert the package to CMsketch
		bool mon = false;
		int p = ss->find(x);
		if (p) mon = true;
		int minv = 0x7fffffff;
		unsigned long long hash[CM_d];
		char* fp=const_cast<char*>(x.c_str());
		for (int i = 0; i < CM_d; i++){
			fp[KEY_LEN]='0'+i;
			hash[i] = bobhash->run(fp, KEY_LEN+1)%(M2-(2*CM_d)+2*i+3);
		}
		for(int i = 0; i < CM_d; i++)
		{
			HK[i][hash[i]].C++;
			minv=min(minv,HK[i][hash[i]].C);
		}
		//use space summary to map the id to a specific value
		if (!mon) 
		{
			if (minv - (ss->getmin()) > 0 || ss->tot < K)
			{
				int i = ss->getid();
				ss->add2(ss->location(x), i);
				ss->str[i] = x;
				ss->sum[i] = minv; 
				ss->link(i, 0);
				while (ss->tot > K)
				{
					int t = ss->Right[0];
					int tmp = ss->head[t];
					ss->cut(ss->head[t]);
					ss->recycling(tmp);
				}
			}
		}
		else{
			if (minv > ss->sum[p])
			{
				int tmp = ss->Left[ss->sum[p]];
				ss->cut(p);
				if (ss->head[ss->sum[p]]) tmp = ss->sum[p];
				ss->sum[p] = minv;
				ss->link(p, tmp);
			}
		}
	}
	
	void work(int n)
	{
        for(int i=N;i;i=ss->Left[i]){
			for(int j=ss->head[i];j;j=ss->Next[j]) 
				{allflowname[ss->str[j]]+=ss->sum[j];}
		}
	        
	}
	
	int merge(int thresh,int opt=false){
		int CNT=0;
		for(unordered_map<string,int>::iterator it=allflowname.begin();it!=allflowname.end();it++){
			q[CNT].x = it->first; q[CNT].y = it->second; 
			CNT++;
    	} 
		sort(q, q + CNT, cmp);
		int bigflow;
		for(bigflow=0;q[bigflow].y>thresh&&bigflow<CNT;bigflow++);
		return bigflow;
	}

	pair<string, int> Query_top(int k)
	{
		return make_pair(q[k].x, q[k].y);
	}

	int Query(string str)
	{
		if(allflowname.find(str) != allflowname.end()){
			return allflowname[str];
		}else{
			return 0;
		}
	}
	
	std::string get_name() {
		return "cmsketch";
	}
};
#endif
