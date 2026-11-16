/* Generated structural sensitivity transport. No approximate sparsity. */
#define NX 15
#define NU 5
#define ND 26
#define NV 277
#define OUT (NX + NX*NX + 2*NX*NU + 2*NX)
static void initialize(double *v, const double *x0) {
    memset(v,0,NV*sizeof(double)); memcpy(v,x0,NX*sizeof(double));
    v[73]=1.0;
    v[92]=1.0;
    v[111]=1.0;
    v[130]=1.0;
    v[149]=1.0;
    v[168]=1.0;
    v[187]=1.0;
    v[206]=1.0;
    v[225]=1.0;
    v[244]=1.0;
}
static int rhs(const double *restrict v, const double *u0, const double *u1,
               double scale, double beta, const double *parameters, double *restrict d) {
    double u[NU], f[NX], jac[ROCKET_NNZ];
    for(int i=0;i<NU;++i) u[i]=u0[i]+beta*(u1[i]-u0[i]);
    rocket_field_prepared(v,u,parameters,f,jac);
    memset(d,0,NV*sizeof(double));
    for(int i=0;i<NX;++i) {if(!isfinite(f[i])) return 2; d[i]=scale*f[i];}
    for(int i=0;i<ROCKET_NNZ;++i) if(!isfinite(jac[i])) return 3;
    d[17]=f[0];
    d[35]=f[1];
    d[53]=f[2];
    d[71]=f[3];
    d[89]=f[4];
    d[107]=f[5];
    d[125]=f[6];
    d[143]=f[7];
    d[161]=f[8];
    d[179]=f[9];
    d[197]=f[10];
    d[215]=f[11];
    d[233]=f[12];
    d[251]=f[13];
    d[276]=f[14];
    { const double a=scale*jac[0];
      d[15]+=a*(1.0-beta);
      d[16]+=a*beta;
    }
    { const double a=scale*jac[1];
      for(int j=0;j<18;++j) d[18+j]+=a*v[72+j];
    }
    { const double a=scale*jac[2];
      for(int j=0;j<18;++j) d[36+j]+=a*v[90+j];
    }
    { const double a=scale*jac[3];
      for(int j=0;j<18;++j) d[54+j]+=a*v[108+j];
    }
    { const double a=scale*jac[4];
      d[72]+=a;
      d[83]+=a*v[15];
      d[86]+=a*v[16];
      d[89]+=a*v[17];
    }
    { const double a=scale*jac[5];
      for(int j=0;j<18;++j) d[72+j]+=a*v[72+j];
    }
    { const double a=scale*jac[6];
      for(int j=0;j<18;++j) d[72+j]+=a*v[90+j];
    }
    { const double a=scale*jac[7];
      for(int j=0;j<18;++j) d[72+j]+=a*v[108+j];
    }
    { const double a=scale*jac[8];
      for(int j=0;j<18;++j) d[72+j]+=a*v[126+j];
    }
    { const double a=scale*jac[9];
      for(int j=0;j<18;++j) d[72+j]+=a*v[144+j];
    }
    { const double a=scale*jac[10];
      for(int j=0;j<18;++j) d[72+j]+=a*v[162+j];
    }
    { const double a=scale*jac[11];
      for(int j=0;j<18;++j) d[72+j]+=a*v[180+j];
    }
    { const double a=scale*jac[12];
      d[83]+=a*(1.0-beta);
      d[86]+=a*beta;
    }
    { const double a=scale*jac[13];
      d[84]+=a*(1.0-beta);
      d[87]+=a*beta;
    }
    { const double a=scale*jac[14];
      d[85]+=a*(1.0-beta);
      d[88]+=a*beta;
    }
    { const double a=scale*jac[15];
      d[90]+=a;
      d[101]+=a*v[15];
      d[104]+=a*v[16];
      d[107]+=a*v[17];
    }
    { const double a=scale*jac[16];
      for(int j=0;j<18;++j) d[90+j]+=a*v[72+j];
    }
    { const double a=scale*jac[17];
      for(int j=0;j<18;++j) d[90+j]+=a*v[90+j];
    }
    { const double a=scale*jac[18];
      for(int j=0;j<18;++j) d[90+j]+=a*v[108+j];
    }
    { const double a=scale*jac[19];
      for(int j=0;j<18;++j) d[90+j]+=a*v[126+j];
    }
    { const double a=scale*jac[20];
      for(int j=0;j<18;++j) d[90+j]+=a*v[144+j];
    }
    { const double a=scale*jac[21];
      for(int j=0;j<18;++j) d[90+j]+=a*v[162+j];
    }
    { const double a=scale*jac[22];
      for(int j=0;j<18;++j) d[90+j]+=a*v[180+j];
    }
    { const double a=scale*jac[23];
      d[101]+=a*(1.0-beta);
      d[104]+=a*beta;
    }
    { const double a=scale*jac[24];
      d[102]+=a*(1.0-beta);
      d[105]+=a*beta;
    }
    { const double a=scale*jac[25];
      d[103]+=a*(1.0-beta);
      d[106]+=a*beta;
    }
    { const double a=scale*jac[26];
      d[108]+=a;
      d[119]+=a*v[15];
      d[122]+=a*v[16];
      d[125]+=a*v[17];
    }
    { const double a=scale*jac[27];
      for(int j=0;j<18;++j) d[108+j]+=a*v[72+j];
    }
    { const double a=scale*jac[28];
      for(int j=0;j<18;++j) d[108+j]+=a*v[90+j];
    }
    { const double a=scale*jac[29];
      for(int j=0;j<18;++j) d[108+j]+=a*v[108+j];
    }
    { const double a=scale*jac[30];
      for(int j=0;j<18;++j) d[108+j]+=a*v[126+j];
    }
    { const double a=scale*jac[31];
      for(int j=0;j<18;++j) d[108+j]+=a*v[144+j];
    }
    { const double a=scale*jac[32];
      for(int j=0;j<18;++j) d[108+j]+=a*v[162+j];
    }
    { const double a=scale*jac[33];
      for(int j=0;j<18;++j) d[108+j]+=a*v[180+j];
    }
    { const double a=scale*jac[34];
      d[119]+=a*(1.0-beta);
      d[122]+=a*beta;
    }
    { const double a=scale*jac[35];
      d[120]+=a*(1.0-beta);
      d[123]+=a*beta;
    }
    { const double a=scale*jac[36];
      d[121]+=a*(1.0-beta);
      d[124]+=a*beta;
    }
    { const double a=scale*jac[37];
      for(int j=0;j<18;++j) d[126+j]+=a*v[144+j];
    }
    { const double a=scale*jac[38];
      for(int j=0;j<18;++j) d[126+j]+=a*v[162+j];
    }
    { const double a=scale*jac[39];
      for(int j=0;j<18;++j) d[126+j]+=a*v[180+j];
    }
    { const double a=scale*jac[40];
      for(int j=0;j<18;++j) d[126+j]+=a*v[198+j];
    }
    { const double a=scale*jac[41];
      for(int j=0;j<18;++j) d[126+j]+=a*v[216+j];
    }
    { const double a=scale*jac[42];
      for(int j=0;j<18;++j) d[126+j]+=a*v[234+j];
    }
    { const double a=scale*jac[43];
      for(int j=0;j<18;++j) d[144+j]+=a*v[126+j];
    }
    { const double a=scale*jac[44];
      for(int j=0;j<18;++j) d[144+j]+=a*v[162+j];
    }
    { const double a=scale*jac[45];
      for(int j=0;j<18;++j) d[144+j]+=a*v[180+j];
    }
    { const double a=scale*jac[46];
      for(int j=0;j<18;++j) d[144+j]+=a*v[198+j];
    }
    { const double a=scale*jac[47];
      for(int j=0;j<18;++j) d[144+j]+=a*v[216+j];
    }
    { const double a=scale*jac[48];
      for(int j=0;j<18;++j) d[144+j]+=a*v[234+j];
    }
    { const double a=scale*jac[49];
      for(int j=0;j<18;++j) d[162+j]+=a*v[126+j];
    }
    { const double a=scale*jac[50];
      for(int j=0;j<18;++j) d[162+j]+=a*v[144+j];
    }
    { const double a=scale*jac[51];
      for(int j=0;j<18;++j) d[162+j]+=a*v[180+j];
    }
    { const double a=scale*jac[52];
      for(int j=0;j<18;++j) d[162+j]+=a*v[198+j];
    }
    { const double a=scale*jac[53];
      for(int j=0;j<18;++j) d[162+j]+=a*v[216+j];
    }
    { const double a=scale*jac[54];
      for(int j=0;j<18;++j) d[162+j]+=a*v[234+j];
    }
    { const double a=scale*jac[55];
      for(int j=0;j<18;++j) d[180+j]+=a*v[126+j];
    }
    { const double a=scale*jac[56];
      for(int j=0;j<18;++j) d[180+j]+=a*v[144+j];
    }
    { const double a=scale*jac[57];
      for(int j=0;j<18;++j) d[180+j]+=a*v[162+j];
    }
    { const double a=scale*jac[58];
      for(int j=0;j<18;++j) d[180+j]+=a*v[198+j];
    }
    { const double a=scale*jac[59];
      for(int j=0;j<18;++j) d[180+j]+=a*v[216+j];
    }
    { const double a=scale*jac[60];
      for(int j=0;j<18;++j) d[180+j]+=a*v[234+j];
    }
    { const double a=scale*jac[61];
      d[198]+=a;
      d[209]+=a*v[15];
      d[212]+=a*v[16];
      d[215]+=a*v[17];
    }
    { const double a=scale*jac[62];
      for(int j=0;j<18;++j) d[198+j]+=a*v[72+j];
    }
    { const double a=scale*jac[63];
      for(int j=0;j<18;++j) d[198+j]+=a*v[90+j];
    }
    { const double a=scale*jac[64];
      for(int j=0;j<18;++j) d[198+j]+=a*v[108+j];
    }
    { const double a=scale*jac[65];
      for(int j=0;j<18;++j) d[198+j]+=a*v[126+j];
    }
    { const double a=scale*jac[66];
      for(int j=0;j<18;++j) d[198+j]+=a*v[144+j];
    }
    { const double a=scale*jac[67];
      for(int j=0;j<18;++j) d[198+j]+=a*v[162+j];
    }
    { const double a=scale*jac[68];
      for(int j=0;j<18;++j) d[198+j]+=a*v[180+j];
    }
    { const double a=scale*jac[69];
      for(int j=0;j<18;++j) d[198+j]+=a*v[198+j];
    }
    { const double a=scale*jac[70];
      for(int j=0;j<18;++j) d[198+j]+=a*v[216+j];
    }
    { const double a=scale*jac[71];
      for(int j=0;j<18;++j) d[198+j]+=a*v[234+j];
    }
    { const double a=scale*jac[72];
      d[209]+=a*(1.0-beta);
      d[212]+=a*beta;
    }
    { const double a=scale*jac[73];
      d[210]+=a*(1.0-beta);
      d[213]+=a*beta;
    }
    { const double a=scale*jac[74];
      d[211]+=a*(1.0-beta);
      d[214]+=a*beta;
    }
    { const double a=scale*jac[75];
      d[216]+=a;
      d[227]+=a*v[15];
      d[230]+=a*v[16];
      d[233]+=a*v[17];
    }
    { const double a=scale*jac[76];
      for(int j=0;j<18;++j) d[216+j]+=a*v[72+j];
    }
    { const double a=scale*jac[77];
      for(int j=0;j<18;++j) d[216+j]+=a*v[90+j];
    }
    { const double a=scale*jac[78];
      for(int j=0;j<18;++j) d[216+j]+=a*v[108+j];
    }
    { const double a=scale*jac[79];
      for(int j=0;j<18;++j) d[216+j]+=a*v[126+j];
    }
    { const double a=scale*jac[80];
      for(int j=0;j<18;++j) d[216+j]+=a*v[144+j];
    }
    { const double a=scale*jac[81];
      for(int j=0;j<18;++j) d[216+j]+=a*v[162+j];
    }
    { const double a=scale*jac[82];
      for(int j=0;j<18;++j) d[216+j]+=a*v[180+j];
    }
    { const double a=scale*jac[83];
      for(int j=0;j<18;++j) d[216+j]+=a*v[198+j];
    }
    { const double a=scale*jac[84];
      for(int j=0;j<18;++j) d[216+j]+=a*v[216+j];
    }
    { const double a=scale*jac[85];
      for(int j=0;j<18;++j) d[216+j]+=a*v[234+j];
    }
    { const double a=scale*jac[86];
      d[227]+=a*(1.0-beta);
      d[230]+=a*beta;
    }
    { const double a=scale*jac[87];
      d[228]+=a*(1.0-beta);
      d[231]+=a*beta;
    }
    { const double a=scale*jac[88];
      d[229]+=a*(1.0-beta);
      d[232]+=a*beta;
    }
    { const double a=scale*jac[89];
      d[234]+=a;
      d[245]+=a*v[15];
      d[248]+=a*v[16];
      d[251]+=a*v[17];
    }
    { const double a=scale*jac[90];
      for(int j=0;j<18;++j) d[234+j]+=a*v[72+j];
    }
    { const double a=scale*jac[91];
      for(int j=0;j<18;++j) d[234+j]+=a*v[90+j];
    }
    { const double a=scale*jac[92];
      for(int j=0;j<18;++j) d[234+j]+=a*v[108+j];
    }
    { const double a=scale*jac[93];
      for(int j=0;j<18;++j) d[234+j]+=a*v[126+j];
    }
    { const double a=scale*jac[94];
      for(int j=0;j<18;++j) d[234+j]+=a*v[144+j];
    }
    { const double a=scale*jac[95];
      for(int j=0;j<18;++j) d[234+j]+=a*v[162+j];
    }
    { const double a=scale*jac[96];
      for(int j=0;j<18;++j) d[234+j]+=a*v[180+j];
    }
    { const double a=scale*jac[97];
      for(int j=0;j<18;++j) d[234+j]+=a*v[198+j];
    }
    { const double a=scale*jac[98];
      for(int j=0;j<18;++j) d[234+j]+=a*v[216+j];
    }
    { const double a=scale*jac[99];
      for(int j=0;j<18;++j) d[234+j]+=a*v[234+j];
    }
    { const double a=scale*jac[100];
      d[245]+=a*(1.0-beta);
      d[248]+=a*beta;
    }
    { const double a=scale*jac[101];
      d[246]+=a*(1.0-beta);
      d[249]+=a*beta;
    }
    { const double a=scale*jac[102];
      d[247]+=a*(1.0-beta);
      d[250]+=a*beta;
    }
    { const double a=scale*jac[103];
      d[253]+=a;
      d[252]+=a*v[18];
      for(int j=0;j<13;++j) d[256+j]+=a*v[19+j];
      for(int j=0;j<3;++j) d[271+j]+=a*v[32+j];
      d[276]+=a*v[35];
    }
    { const double a=scale*jac[104];
      d[254]+=a;
      d[252]+=a*v[36];
      for(int j=0;j<13;++j) d[256+j]+=a*v[37+j];
      for(int j=0;j<3;++j) d[271+j]+=a*v[50+j];
      d[276]+=a*v[53];
    }
    { const double a=scale*jac[105];
      d[255]+=a;
      d[252]+=a*v[54];
      for(int j=0;j<13;++j) d[256+j]+=a*v[55+j];
      for(int j=0;j<3;++j) d[271+j]+=a*v[68+j];
      d[276]+=a*v[71];
    }
    { const double a=scale*jac[106];
      d[252]+=a*v[72];
      for(int j=0;j<13;++j) d[256+j]+=a*v[73+j];
      for(int j=0;j<3;++j) d[271+j]+=a*v[86+j];
      d[276]+=a*v[89];
    }
    { const double a=scale*jac[107];
      d[252]+=a*v[90];
      for(int j=0;j<13;++j) d[256+j]+=a*v[91+j];
      for(int j=0;j<3;++j) d[271+j]+=a*v[104+j];
      d[276]+=a*v[107];
    }
    { const double a=scale*jac[108];
      d[252]+=a*v[108];
      for(int j=0;j<13;++j) d[256+j]+=a*v[109+j];
      for(int j=0;j<3;++j) d[271+j]+=a*v[122+j];
      d[276]+=a*v[125];
    }
    { const double a=scale*jac[109];
      d[252]+=a*v[126];
      for(int j=0;j<13;++j) d[256+j]+=a*v[127+j];
      for(int j=0;j<3;++j) d[271+j]+=a*v[140+j];
      d[276]+=a*v[143];
    }
    { const double a=scale*jac[110];
      d[252]+=a*v[144];
      for(int j=0;j<13;++j) d[256+j]+=a*v[145+j];
      for(int j=0;j<3;++j) d[271+j]+=a*v[158+j];
      d[276]+=a*v[161];
    }
    { const double a=scale*jac[111];
      d[252]+=a*v[162];
      for(int j=0;j<13;++j) d[256+j]+=a*v[163+j];
      for(int j=0;j<3;++j) d[271+j]+=a*v[176+j];
      d[276]+=a*v[179];
    }
    { const double a=scale*jac[112];
      d[252]+=a*v[180];
      for(int j=0;j<13;++j) d[256+j]+=a*v[181+j];
      for(int j=0;j<3;++j) d[271+j]+=a*v[194+j];
      d[276]+=a*v[197];
    }
    { const double a=scale*jac[113];
      d[252]+=a*v[198];
      for(int j=0;j<13;++j) d[256+j]+=a*v[199+j];
      for(int j=0;j<3;++j) d[271+j]+=a*v[212+j];
      d[276]+=a*v[215];
    }
    { const double a=scale*jac[114];
      d[252]+=a*v[216];
      for(int j=0;j<13;++j) d[256+j]+=a*v[217+j];
      for(int j=0;j<3;++j) d[271+j]+=a*v[230+j];
      d[276]+=a*v[233];
    }
    { const double a=scale*jac[115];
      d[252]+=a*v[234];
      for(int j=0;j<13;++j) d[256+j]+=a*v[235+j];
      for(int j=0;j<3;++j) d[271+j]+=a*v[248+j];
      d[276]+=a*v[251];
    }
    { const double a=scale*jac[116];
      d[266]+=a*(1.0-beta);
      d[271]+=a*beta;
    }
    { const double a=scale*jac[117];
      d[267]+=a*(1.0-beta);
      d[272]+=a*beta;
    }
    { const double a=scale*jac[118];
      d[269]+=a*(1.0-beta);
      d[274]+=a*beta;
    }
    { const double a=scale*jac[119];
      d[270]+=a*(1.0-beta);
      d[275]+=a*beta;
    }
    return 0;
}
static void unpack(const double *v,const double *x0,const double *u0,const double *u1,
                   double scale,int free_time,double *out) {
    memset(out,0,OUT*sizeof(double));memcpy(out,v,NX*sizeof(double));
    out[15]=1.0;
    out[240]=v[15];
    out[315]=v[16];
    out[390]=v[17];
    out[30]=v[18];
    out[31]=1.0;
    out[34]=v[19];
    out[35]=v[20];
    out[36]=v[21];
    out[37]=v[22];
    out[38]=v[23];
    out[39]=v[24];
    out[40]=v[25];
    out[41]=v[26];
    out[42]=v[27];
    out[43]=v[28];
    out[245]=v[29];
    out[246]=v[30];
    out[247]=v[31];
    out[320]=v[32];
    out[321]=v[33];
    out[322]=v[34];
    out[391]=v[35];
    out[45]=v[36];
    out[47]=1.0;
    out[49]=v[37];
    out[50]=v[38];
    out[51]=v[39];
    out[52]=v[40];
    out[53]=v[41];
    out[54]=v[42];
    out[55]=v[43];
    out[56]=v[44];
    out[57]=v[45];
    out[58]=v[46];
    out[250]=v[47];
    out[251]=v[48];
    out[252]=v[49];
    out[325]=v[50];
    out[326]=v[51];
    out[327]=v[52];
    out[392]=v[53];
    out[60]=v[54];
    out[63]=1.0;
    out[64]=v[55];
    out[65]=v[56];
    out[66]=v[57];
    out[67]=v[58];
    out[68]=v[59];
    out[69]=v[60];
    out[70]=v[61];
    out[71]=v[62];
    out[72]=v[63];
    out[73]=v[64];
    out[255]=v[65];
    out[256]=v[66];
    out[257]=v[67];
    out[330]=v[68];
    out[331]=v[69];
    out[332]=v[70];
    out[393]=v[71];
    out[75]=v[72];
    out[79]=v[73];
    out[80]=v[74];
    out[81]=v[75];
    out[82]=v[76];
    out[83]=v[77];
    out[84]=v[78];
    out[85]=v[79];
    out[86]=v[80];
    out[87]=v[81];
    out[88]=v[82];
    out[260]=v[83];
    out[261]=v[84];
    out[262]=v[85];
    out[335]=v[86];
    out[336]=v[87];
    out[337]=v[88];
    out[394]=v[89];
    out[90]=v[90];
    out[94]=v[91];
    out[95]=v[92];
    out[96]=v[93];
    out[97]=v[94];
    out[98]=v[95];
    out[99]=v[96];
    out[100]=v[97];
    out[101]=v[98];
    out[102]=v[99];
    out[103]=v[100];
    out[265]=v[101];
    out[266]=v[102];
    out[267]=v[103];
    out[340]=v[104];
    out[341]=v[105];
    out[342]=v[106];
    out[395]=v[107];
    out[105]=v[108];
    out[109]=v[109];
    out[110]=v[110];
    out[111]=v[111];
    out[112]=v[112];
    out[113]=v[113];
    out[114]=v[114];
    out[115]=v[115];
    out[116]=v[116];
    out[117]=v[117];
    out[118]=v[118];
    out[270]=v[119];
    out[271]=v[120];
    out[272]=v[121];
    out[345]=v[122];
    out[346]=v[123];
    out[347]=v[124];
    out[396]=v[125];
    out[120]=v[126];
    out[124]=v[127];
    out[125]=v[128];
    out[126]=v[129];
    out[127]=v[130];
    out[128]=v[131];
    out[129]=v[132];
    out[130]=v[133];
    out[131]=v[134];
    out[132]=v[135];
    out[133]=v[136];
    out[275]=v[137];
    out[276]=v[138];
    out[277]=v[139];
    out[350]=v[140];
    out[351]=v[141];
    out[352]=v[142];
    out[397]=v[143];
    out[135]=v[144];
    out[139]=v[145];
    out[140]=v[146];
    out[141]=v[147];
    out[142]=v[148];
    out[143]=v[149];
    out[144]=v[150];
    out[145]=v[151];
    out[146]=v[152];
    out[147]=v[153];
    out[148]=v[154];
    out[280]=v[155];
    out[281]=v[156];
    out[282]=v[157];
    out[355]=v[158];
    out[356]=v[159];
    out[357]=v[160];
    out[398]=v[161];
    out[150]=v[162];
    out[154]=v[163];
    out[155]=v[164];
    out[156]=v[165];
    out[157]=v[166];
    out[158]=v[167];
    out[159]=v[168];
    out[160]=v[169];
    out[161]=v[170];
    out[162]=v[171];
    out[163]=v[172];
    out[285]=v[173];
    out[286]=v[174];
    out[287]=v[175];
    out[360]=v[176];
    out[361]=v[177];
    out[362]=v[178];
    out[399]=v[179];
    out[165]=v[180];
    out[169]=v[181];
    out[170]=v[182];
    out[171]=v[183];
    out[172]=v[184];
    out[173]=v[185];
    out[174]=v[186];
    out[175]=v[187];
    out[176]=v[188];
    out[177]=v[189];
    out[178]=v[190];
    out[290]=v[191];
    out[291]=v[192];
    out[292]=v[193];
    out[365]=v[194];
    out[366]=v[195];
    out[367]=v[196];
    out[400]=v[197];
    out[180]=v[198];
    out[184]=v[199];
    out[185]=v[200];
    out[186]=v[201];
    out[187]=v[202];
    out[188]=v[203];
    out[189]=v[204];
    out[190]=v[205];
    out[191]=v[206];
    out[192]=v[207];
    out[193]=v[208];
    out[295]=v[209];
    out[296]=v[210];
    out[297]=v[211];
    out[370]=v[212];
    out[371]=v[213];
    out[372]=v[214];
    out[401]=v[215];
    out[195]=v[216];
    out[199]=v[217];
    out[200]=v[218];
    out[201]=v[219];
    out[202]=v[220];
    out[203]=v[221];
    out[204]=v[222];
    out[205]=v[223];
    out[206]=v[224];
    out[207]=v[225];
    out[208]=v[226];
    out[300]=v[227];
    out[301]=v[228];
    out[302]=v[229];
    out[375]=v[230];
    out[376]=v[231];
    out[377]=v[232];
    out[402]=v[233];
    out[210]=v[234];
    out[214]=v[235];
    out[215]=v[236];
    out[216]=v[237];
    out[217]=v[238];
    out[218]=v[239];
    out[219]=v[240];
    out[220]=v[241];
    out[221]=v[242];
    out[222]=v[243];
    out[223]=v[244];
    out[305]=v[245];
    out[306]=v[246];
    out[307]=v[247];
    out[380]=v[248];
    out[381]=v[249];
    out[382]=v[250];
    out[403]=v[251];
    out[225]=v[252];
    out[226]=v[253];
    out[227]=v[254];
    out[228]=v[255];
    out[229]=v[256];
    out[230]=v[257];
    out[231]=v[258];
    out[232]=v[259];
    out[233]=v[260];
    out[234]=v[261];
    out[235]=v[262];
    out[236]=v[263];
    out[237]=v[264];
    out[238]=v[265];
    out[239]=1.0;
    out[310]=v[266];
    out[311]=v[267];
    out[312]=v[268];
    out[313]=v[269];
    out[314]=v[270];
    out[385]=v[271];
    out[386]=v[272];
    out[387]=v[273];
    out[388]=v[274];
    out[389]=v[275];
    out[404]=v[276];
    for(int i=0;i<NX;++i) {
        double affine=v[i];
        for(int j=0;j<NX;++j) affine-=out[NX+i*NX+j]*x0[j];
        for(int j=0;j<NU;++j) affine-=out[NX+NX*NX+i*NU+j]*u0[j];
        for(int j=0;j<NU;++j) affine-=out[NX+NX*NX+NX*NU+i*NU+j]*u1[j];
        if(free_time) affine-=out[NX+NX*NX+2*NX*NU+i]*scale;
        out[OUT-NX+i]=affine;
    }
}
