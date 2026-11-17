"""patch_hid.py <kimi_k3.c>: K3_HID_DUMP, a measurement dump for a draft model's acceptance."""
import io, sys

p = sys.argv[1]
s = io.open(p, newline='', encoding='utf-8').read()
nl = '\r\n' if '\r\n' in s else '\n'
s = s.replace('\r\n', '\n')
if 'K3_HID_DUMP' in s:
    print('already patched'); sys.exit(0)


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)


rep('''static FILE *g_lfp=NULL;                       /* K3_LOGITS: per-position logit dump */
''', '''static FILE *g_lfp=NULL;                       /* K3_LOGITS: per-position logit dump */
/* K3_HID_DUMP=<prefix> (a measurement, for a draft model's acceptance): the rows after the
 * layers of K3_HID_LAYERS (0-based, comma separated) appended to <prefix>.L<layer>.f32 in the
 * order the steps run, each step's first position and row count to <prefix>.steps.txt, every
 * position's input id to <prefix>.ids.i32, and the id the head ranks first after it with its
 * margin over the second to <prefix>.top1.i32 and .margin.f32 (the head then runs at every row). */
static FILE *g_hid_f[160], *g_hid_ids, *g_hid_top, *g_hid_mar, *g_hid_steps;
static int g_hid_state=-1;
static int k3_hid_on(void){
    if(g_hid_state<0){
        const char *pf=getenv("K3_HID_DUMP"), *ls=getenv("K3_HID_LAYERS");
        g_hid_state=0;
        if(pf&&*pf&&ls&&*ls){
            char nm[1024]; int n=0;
            for(const char *q=ls;*q;){
                int li=atoi(q);
                if(li>=0&&li<160&&!g_hid_f[li]){ snprintf(nm,sizeof nm,"%s.L%02d.f32",pf,li); g_hid_f[li]=fopen(nm,"ab"); n+=g_hid_f[li]!=NULL; }
                while(*q&&*q!=',') q++;
                if(*q==',') q++;
            }
            snprintf(nm,sizeof nm,"%s.ids.i32",pf);    g_hid_ids=fopen(nm,"ab");
            snprintf(nm,sizeof nm,"%s.top1.i32",pf);   g_hid_top=fopen(nm,"ab");
            snprintf(nm,sizeof nm,"%s.margin.f32",pf); g_hid_mar=fopen(nm,"ab");
            snprintf(nm,sizeof nm,"%s.steps.txt",pf);  g_hid_steps=fopen(nm,"a");
            g_hid_state=n>0&&g_hid_ids&&g_hid_top&&g_hid_mar&&g_hid_steps;
            fprintf(stderr,"[K3] K3_HID_DUMP: %d layers' rows, the ids and the head's first id at every position to %s.* (%s)\\n",
                    n,pf,g_hid_state?"on":"cannot open the files: off");
        }
    }
    return g_hid_state;
}
''')

rep('''        if(g_k3_ckf){ k3_ck_rows(K3CK_HID,i,hidden,C,D); fflush(g_k3_ckf); }
''', '''        if(g_k3_ckf){ k3_ck_rows(K3CK_HID,i,hidden,C,D); fflush(g_k3_ckf); }
        if(k3_hid_on()&&i<160&&g_hid_f[i]){ fwrite(hidden,sizeof(float),(size_t)C*D,g_hid_f[i]); fflush(g_hid_f[i]); }
''')

rep('''    k3_embed(m,ids,pos0,C,hidden);
    if(pos0==0&&C<=1) fprintf(stderr,"[DBG] step_chunk''', '''    k3_embed(m,ids,pos0,C,hidden);
    if(k3_hid_on()){ fwrite(ids,sizeof(int),(size_t)C,g_hid_ids); fprintf(g_hid_steps,"%d %d\\n",pos0,C); fflush(g_hid_ids); fflush(g_hid_steps); }
    if(pos0==0&&C<=1) fprintf(stderr,"[DBG] step_chunk''')

rep('''            if(!g_lfp && !g_k3_val_lfp && !echo_on && t<C-1) continue;
''', '''            if(!g_lfp && !g_k3_val_lfp && !echo_on && !k3_hid_on() && t<C-1) continue;
''')

rep('''            if(g_lfp) fwrite(lo,sizeof(float),(size_t)c->vocab,g_lfp);
''', '''            if(k3_hid_on()){
                int b=0, b2=-1;
                for(int v=1;v<c->vocab;v++){
                    if(lo[v]>lo[b]){ b2=b; b=v; }
                    else if(b2<0||lo[v]>lo[b2]) b2=v;
                }
                float mg=lo[b]-lo[b2];
                fwrite(&b,sizeof(int),1,g_hid_top); fwrite(&mg,sizeof(float),1,g_hid_mar);
                if(t==C-1){ fflush(g_hid_top); fflush(g_hid_mar); }
            }
            if(g_lfp) fwrite(lo,sizeof(float),(size_t)c->vocab,g_lfp);
''')

io.open(p, 'w', newline='', encoding='utf-8').write(s.replace('\n', nl))
print('patched', p)
