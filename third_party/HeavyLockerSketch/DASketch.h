#ifndef _dasketch_H
#define _dasketch_H
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <string>
#include <cstring>
#include <climits>
#include <unordered_map>
#include "BaseSketch.h"
#include "BOBHASH32.h"
#include "params.h"
#include "BOBHASH64.h"
#define CMM_d 1
#define TOP_d 4
#define opt1 0 //define the query method
#define rep(i,a,n) for(int i=a;i<=n;i++)
using namespace std;
class dasketch : public sketch::BaseSketch{
private:
    struct cell {
        string ID;
        int Cs;
        int Cr;
    };
    struct bucket { 
        cell cells[TOP_d];
    };
    bucket *bk; 
    struct node { int C; } HK[CMM_d][MAX_MEM+10];
    BOBHash64 * bobhash; 
    int M2; 
    int total;
public:
    dasketch(int M2) :M2(M2) { 
        bk = new bucket[M2]; 
        bobhash = new BOBHash64(1005); 
        total = 0; 
    }
    void clear()
    {
        for (int i = 0; i < M2; i++) 
            for (int j = 0; j < TOP_d; j++) {
                bk[i].cells[j].ID = "";
                bk[i].cells[j].Cs = 0;
                bk[i].cells[j].Cr = 0;
            }
        for (int i = 0; i < CMM_d; i++) 
            for (int j = 0; j <= M2 + 5; j++)
                HK[i][j].C = 0;
        total = 0; 
    }

    unsigned long long Hash(string ST)
    {
        return (bobhash->run(ST.c_str(), ST.size()));
    }

    void Insert(const string x)
    {
        char* fp=const_cast<char*>(x.c_str());
        int h = bobhash->run(fp, KEY_LEN)%M2; 
        bool match = false; 
        bool empty = false; 
        int minv= 0x7fffffff; 
        int minp = -1; 
        for (int i = 0; i < TOP_d; i++) { 
            if (bk[h].cells[i].ID == x) { 
                bk[h].cells[i].Cs++;
                bk[h].cells[i].Cr++;
                match = true;
                break;
            }
        }
        if (!match) { 
            for (int i = 0; i < TOP_d; i++) {
                if (bk[h].cells[i].ID == "") { 
                    bk[h].cells[i].ID = x;
                    bk[h].cells[i].Cs = 1;
                    bk[h].cells[i].Cr = 1;
                    empty = true;
                    break;
                }
            }
        }
        if (!match && !empty) { 
            for(int i = 0; i < TOP_d; i++){ 
			    if(bk[h].cells[i].Cs<minv){
                    minv = bk[h].cells[i].Cs;
                    minp = i;
                }
		    }
            double p = 1.0 / (minv + 1); 
            double r = rand() / double(RAND_MAX); 
            if (r < p) { 
                string evicted = bk[h].cells[minp].ID; 
                int evicted_cr = bk[h].cells[minp].Cr;
                bk[h].cells[minp].ID = x;
                bk[h].cells[minp].Cs = minv + 1; 
                bk[h].cells[minp].Cr = 1; 
                unsigned long long hash[CMM_d]; 
                for (int i = 0; i < CMM_d; i++){
                    char* fp=const_cast<char*>(evicted.c_str());
                    fp[KEY_LEN]='0'+i;
                    hash[i] = bobhash->run(fp, KEY_LEN+1)%(M2-(2*CMM_d)+2*i+3); 
                }
                    
                for(int i = 0; i < CMM_d; i++) { 
                    HK[i][hash[i]].C += evicted_cr;
                }
                total += evicted_cr; 
            }
            else { 
                unsigned long long hash[CMM_d]; 
                for (int i = 0; i < CMM_d; i++){
                    char* fp=const_cast<char*>(x.c_str());
                    fp[KEY_LEN]='0'+i;
                    hash[i] = bobhash->run(fp, KEY_LEN+1)%(M2-(2*CMM_d)+2*i+3); 
                }
                    
                for(int i = 0; i < CMM_d; i++) { 
                    HK[i][hash[i]].C++;
                }
                total++; 
            }
        }
    }

    void work(int n)
    {
        for(int i=0;i<TOP_d;i++){
            for(int j= 0; j < M2; j++){
                mergename[n][i][j]=bk[j].cells[i].ID;
                mergeresult1[n][i][j]=bk[j].cells[i].Cs;
                mergeresult2[n][i][j]=bk[j].cells[i].Cr;
            }
        }
        for(int i=0;i<CMM_d;i++){
            for(int j= 0; j < M2; j++){
                mergeresult3[n][i][j]=HK[i][j].C;
            }
        }
        totalnum[n]=total;
    }

	int merge(int thresh,int opt=0){
        int CNT = 0;
		for(unordered_map<string,int>::iterator it=allflowname.begin();it!=allflowname.end();it++){
			it->second=0;
    	}
		for(int i=0;i<M2;i++){
			int cnt=0;
			unordered_map<string,int> temp;
			for(int z=0;z<node_num;z++){
				for(int j=0;j<TOP_d;j++){
					if(temp.find(mergename[z][j][i]) != temp.end()){
						temp[mergename[z][j][i]]+=mergeresult1[z][j][i];
					}else{
						temp[mergename[z][j][i]]=mergeresult1[z][j][i];
					} 	
                    allflowname[mergename[z][j][i]]=mergeresult2[z][j][i];
				}
			}
			for(unordered_map<string,int>::iterator it=temp.begin();it!=temp.end();it++){
				q[cnt].x=it->first;
				q[cnt].y=it->second;
				cnt++;
			}
			sort(q,q+cnt,cmp);
			for(int j=0;j<TOP_d;j++){
				bk[i].cells[j].Cs=q[j].y;
                bk[i].cells[j].ID=q[j].x;
                bk[i].cells[j].Cr=allflowname[q[j].x];
			}
		}
		if(opt==false){
            for(int i=1;i<node_num;i++){
                totalnum[0]=max(totalnum[i],totalnum[0]);
            }
			for(int i=0;i<M2;i++){
                for(int j=0;j<CMM_d;j++){
                    int maxv=-1;
                    for(int z=0;z<node_num;z++){
					    maxv=max(maxv,mergeresult3[z][j][i]);
				    }
                    HK[j][i].C=maxv;
                }
			}
		}else{
            for(int i=1;i<node_num;i++){
                totalnum[0]+=totalnum[i];
            }
			for(int i=0;i<M2;i++){
                for(int j=0;j<CMM_d;j++){
                    int maxv=0;
                    for(int z=0;z<node_num;z++){
					    maxv+=mergeresult3[z][j][i];
				    }
                    HK[j][i].C=maxv;
                }
			}	
		}

        for(unordered_map<string,int>::iterator it=allflowname.begin();it!=allflowname.end();it++){
            int h = Hash(it->first) % M2;
            int result=0;
            unsigned long long hash[CMM_d]; 
            if(opt1==1){ //get estimated value from heavy part and light part
                int temp=0;
                for (int j = 0; j < TOP_d; j++) {
                    if(bk[h].cells[j].ID==it->first){
                        temp=bk[h].cells[j].Cr;
                    }
                }
                int light_num=INT_MAX;
                for(int j=0;j<CMM_d;j++){
                    hash[j]=Hash(it->first+std::to_string(j))%(M2-(2*CMM_d)+2*j+3);
                    light_num=min(light_num,HK[j][hash[j]].C);
                }
                light_num=light_num-CMM_d * (totalnum[0] - light_num) / (M2 - 1);
                result=temp+light_num;
            }else{      //only query the heavy part
                for (int j = 0; j < TOP_d; j++) {
                    if(bk[h].cells[j].ID==it->first){
                        result=bk[h].cells[j].Cs;
                    }
                }
            }
            it->second=result;
    	}
		for(unordered_map<string,int>::iterator it=allflowname.begin();it!=allflowname.end();it++){
			q[CNT].x = it->first; q[CNT].y = it->second; CNT++;
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
        return "dasketch";
    }
};
#endif 