#ifndef _hyperuss_H
#define _hyperuss_H
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <string>
#include <cstring>
#include "BaseSketch.h"
#include "BOBHASH64.h"
#include "params.h"
#define HU_d 4
#define rep(i,a,n) for(int i=a;i<=n;i++)
using namespace std;
class hyperuss : public sketch::BaseSketch{
private:
	struct node { string ID; int C; } HK[HU_d][MAX_MEM+10];
	BOBHash64 * bobhash;
	int M2;
public:
	hyperuss(int M2) :M2(M2){ bobhash = new BOBHash64(1005); }

	void clear()	{
		for (int i = 0; i < HU_d; i++)
			for (int j = 0; j <= M2 + 5; j++)
				HK[i][j].C = 0,HK[i][j].ID ="";
	}

	unsigned long long Hash(string ST)	{
		return (bobhash->run(ST.c_str(), ST.size()));
	}

	void Insert(const string x)	{
		int minv = 0x7fffffff;
		unsigned long long hash[HU_d];
		char* fp=const_cast<char*>(x.c_str());
		for (int i = 0; i < HU_d; i++){
            fp[KEY_LEN]='0'+i;
			hash[i] = bobhash->run(fp, KEY_LEN+1)%(M2-(2*HU_d)+2*i+3);
		}
			
        bool flag0 = false,flag1 = false;
        for(int i = 0; i < HU_d; i++)
		{
			if (HK[i][hash[i]].ID == x) {
				HK[i][hash[i]].C++;
                flag0 = true;
                break;
			}
        }
        if(!flag0){                                                                                             
            for(int i = 0; i < HU_d; i++){
			    if (HK[i][hash[i]].ID == "") {
				    HK[i][hash[i]].ID=x;
				    HK[i][hash[i]].C=1;
                    flag1 = true;
                    break;
			    }
            }
        }
        if(!flag0 && !flag1){
            int mini;
            for(int i=0;i<HU_d;i++){
                if(HK[i][hash[i]].C<minv){
                    minv = HK[i][hash[i]].C;
                    mini = i;
                }
            }
            double p = 1.0 / (minv + 1);
			double q = 1.0 - p;
			double r = (double)rand() / RAND_MAX;
			if (r < p) {
				HK[mini][hash[mini]].ID = x;
				HK[mini][hash[mini]].C =1/p;
			}
			else {
				HK[mini][hash[mini]].C = minv / q;
			}
        }
	}
	 
    void work(int n)
	{
		int CNT = 0;
		for (int i = 0; i<HU_d; i++){
			for(int j=0;j<M2;j++){
				if(HK[i][j].ID!=""){
					mergename[n][i][j] = HK[i][j].ID; 
					mergeresult1[n][i][j] = HK[i][j].C; 
				}else{
					mergename[n][i][j] =""; 
					mergeresult1[n][i][j] =0; 
				}
			}		
		}
	}

	int merge(int thresh,int opt=0){
		int CNT=0;
		unsigned long long hash[HU_d];
		for(unordered_map<string,int>::iterator it=allflowname.begin();it!=allflowname.end();it++){
			it->second=0;
    	}
		for (int i = 0; i<HU_d; i++){
			for(int j=0;j<M2;j++){
				int maxnum=-1;
				string maxid="";
				unordered_map<string,int> temp;
				for(int k=0;k<node_num;k++){
					if(temp.find(mergename[k][i][j]) != temp.end()){
						temp[mergename[k][i][j]]+=mergeresult1[k][i][j];
					}else{
						temp[mergename[k][i][j]]=mergeresult1[k][i][j];
					} 	
				}
				for(unordered_map<string,int>::iterator it=temp.begin();it!=temp.end();it++){
					if(maxnum<it->second){
						maxnum=it->second;
						maxid=it->first;
					}
				}
				HK[i][j].ID=maxid;
				HK[i][j].C=maxnum;
			}		
		}
		for(unordered_map<string,int>::iterator it=allflowname.begin();it!=allflowname.end();it++){
			char* fp=const_cast<char*>(it->first.c_str());
			int result=0;
			for (int i = 0; i < HU_d; i++){
            	fp[KEY_LEN]='0'+i;
				hash[i] = bobhash->run(fp, KEY_LEN+1)%(M2-(2*HU_d)+2*i+3);
				if(HK[i][hash[i]].ID==it->first){
					result+=HK[i][hash[i]].C;
				}
			}
			it->second=result;
			q[CNT].x = it->first; q[CNT].y = result; CNT++;
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
		return "hyperuss";
	}
};
#endif
