/* k3_sessions.h -- several conversations kept beside the one the engine holds (K3_SESSIONS),
 * in RAM and in files (K3_SESSION_DIR). Included once by kimi_k3.c, after the photos and the
 * short request's park: it uses both, and the Model, K3Seq and the caches' accounting.
 *
 * THE PROBLEM. The engine holds one conversation: the KDA layers' recurrent state and the MLA
 * layers' rows. A request that belongs to another conversation -- a second chat window, an
 * agent's sub-agent with instructions of its own, a client's summary request -- wrote its rows
 * over the first one's, and the first conversation's next turn read every token again: 221 s
 * for 16.5 thousand tokens on Kimi K3.1 Pro, a quarter of an hour for 60 thousand. A restart of
 * the server cost every conversation the same.
 *
 * THE IDEA. A conversation's state is a set of arrays (K3Seq), and the Model holds pointers to
 * them. When a request is another conversation's, the one held goes aside whole -- the pointers
 * are traded, nothing is copied -- and comes back the same way when a request continues it.
 * The new conversation starts in rows of its own; what it shares with a kept one up to a photo
 * of the pool (the system prompt's end) is copied over, so it still starts after it.
 *
 * WHICH CONVERSATION A REQUEST IS. The live conversation and the kept ones are each asked how
 * many of the prompt's tokens they save (k3_sess_gain). A prompt that continues a conversation,
 * or parts from it only past the end of its first record that is not the system's (`key`: a
 * regenerated reply, an edited turn), is that conversation's: its rows, the photo kept with it
 * (its last prompt's end, where a client that drops the reasoning comes back) or one of the
 * pool's. A prompt that parts before `key` is another conversation, which a kept one can only
 * lend the rows of what they share up to one of the pool's photos. The best becomes the live
 * conversation, or the live one goes aside and the new one starts after what it was lent.
 *
 * RAM, AND A LONG REASONING. A kept conversation's RAM would come out of the expert cache, as
 * the live conversation's rows past the plan do (K3_LEND): fewer experts in RAM, every token
 * of every reply a little slower. With files there is no reason to pay that -- a conversation
 * is read back from its file in a second or a few, where its tokens took minutes or hours --
 * so with K3_SESSION_DIR a conversation that goes aside is written and leaves RAM at once
 * (K3_SESSION_GB=0, the default there), and the expert cache is exactly what it is without
 * this file. Without files the kept ones and the live one's rows past the plan share
 * K3_SESSION_GB (8): as the live conversation grows, the kept ones leave, the longest unused
 * first (k3_sess_yield, where the caches grow), and past 8 GB of its rows (120 thousand
 * positions) none is left. Nothing here runs per token: a conversation is traded, read or
 * written only when a request starts, when the server has been idle, and when it exits.
 *
 * FILES. NNNNNNNN.k3r holds a conversation's rows, a position after the other (a later save
 * appends, and a turn written again overwrites from where it parted); NNNNNNNN.k3s its tokens,
 * the KDA state at its end and the photo below it. Written when the conversation goes aside,
 * when the server has had no request for K3_SESSION_IDLE seconds, and at exit; never while a
 * reply is computed. The record goes first and comes last (renamed into place once the rows
 * are on the drive), so a file cut short by a crash is no conversation. The directory is read
 * at the start; a file is trusted only for the model, the quantization and the format it was
 * written by, and what is not trusted is deleted. K3_SESSION_DISK_GB bounds the files, the
 * longest unused conversation's going first.
 *
 * Off with K3_SESSIONS unset or 0, in a multiplexed serve (KV_SLOTS > 1) and under Metal. */
#ifndef K3_SESSIONS_H
#define K3_SESSIONS_H

#include <dirent.h>

#define K3_SESS_MAX 64
#define K3_SESS_MAGIC 0x31305353334B4F43ull   /* "COK3SS01" */
#define K3_SESS_END   0x444E453153334B43ull   /* "CK3S1END" */
#define K3_SESS_VERSION 1                     /* raise when the rows' or the state's bytes change */

typedef struct {
    uint64_t magic, fp, id, stamp, state_floats;
    int32_t len, ppos, plog, vocab, rowbytes, pad;
} K3SessHead;

typedef struct {
    int on, ram;                  /* the entry holds a conversation; its state is in RAM (q) */
    uint64_t id;                  /* its files' number, 0: none */
    int saved; double dbytes;     /* the positions its rows file holds; the files' bytes */
    int *fed, len;                /* its tokens */
    K3Seq q;                      /* ram: its state */
    int ppos, plog;               /* the photo kept with it (its last prompt's end), with logits or not */
    float *pblob, *plogit;        /* ram: that photo */
    uint64_t used;
} K3Sess;
static K3Sess g_k3_sess[K3_SESS_MAX];
static int g_k3_sess_n, g_k3_sess_min=1024, g_k3_sess_idle=30;
static double g_k3_sess_ram=8e9, g_k3_sess_disk=64e9;
static char g_k3_sess_dir[900];
static uint64_t g_k3_sess_clock, g_k3_sess_next=1, g_k3_sess_fp;
static uint64_t g_k3_live_id;     /* the live conversation's files, 0: none */
static double g_k3_live_dbytes;

static int k3_sess_on(void){ return g_k3_sess_n>0&&g_k3_mux_slots==1; }
static double k3_state_bytes(const Model *m){ return (double)k3_ckpt_blob_floats(m)*sizeof(float); }
static int k3_common(const int *a,int na,const int *b,int nb){
    const int n=na<nb?na:nb; int i=0;
    while(i<n&&a[i]==b[i]) i++;
    return i;
}
/* the bytes of a conversation's rows, as kv_array_to counted them */
static double k3_seq_rows(const Model *m,const K3Seq *q){
    const Cfg *c=&m->c; double b=0.0;
    for(int i=0;i<c->n_layers;i++){
        if(q->Lc&&q->Lc[i]) b+=(double)q->kv_cap*c->kv_lora*sizeof(float);
        if(q->Rc&&q->Rc[i]) b+=(double)q->kv_cap*c->qk_rope*sizeof(float);
        if(q->Ic&&q->Ic[i]) b+=(double)q->kv_cap*c->index_hd*sizeof(float);
    }
    return b;
}
static void k3_seq_free(Model *m,K3Seq *q){
    const Cfg *c=&m->c;
    g_k3_kv_bytes-=k3_seq_rows(m,q);
    if(g_k3_kv_bytes<0.0) g_k3_kv_bytes=0.0;
    for(int i=0;i<c->n_layers;i++){
        if(q->kstate) free(q->kstate[i]);
        if(q->cwq) free(q->cwq[i]);
        if(q->cwk) free(q->cwk[i]);
        if(q->cwv) free(q->cwv[i]);
        if(q->Lc) free(q->Lc[i]);
        if(q->Rc) free(q->Rc[i]);
        if(q->Ic) free(q->Ic[i]);
    }
    free(q->kstate); free(q->cwq); free(q->cwk); free(q->cwv); free(q->Lc); free(q->Rc); free(q->Ic);
    kv_prefix_free(&q->kvp);
    memset(q,0,sizeof *q);
}
/* what a kept conversation holds in RAM beside its rows: the KDA state, its photo */
static double k3_sess_extra_of(const Model *m,const K3Sess *s){
    return k3_state_bytes(m)*(s->pblob?2.0:1.0)+(s->plogit?(double)m->c.vocab*sizeof(float):0.0);
}

/* ---- the files ---- */
static void k3_sess_path(char *out,size_t n,uint64_t id,const char *ext){
    snprintf(out,n,"%s/%08llu.%s",g_k3_sess_dir,(unsigned long long)id,ext);
}
static void k3_sess_unlink(uint64_t id){
    char p[1024];
    k3_sess_path(p,sizeof p,id,"k3s"); remove(p);
    k3_sess_path(p,sizeof p,id,"k3s.tmp"); remove(p);
    k3_sess_path(p,sizeof p,id,"k3r"); remove(p);
}
static int64_t k3_sess_fsize(const char *path){
    int fd=open(path,COMPAT_O_RDONLY);
    if(fd<0) return -1;
    const int64_t n=k3dc_fsize(fd);
    close(fd);
    return n;
}
static int k3_sess_seek(FILE *f,int64_t off){
#ifdef _WIN32
    return _fseeki64(f,off,SEEK_SET)==0;
#else
    return fseeko(f,(off_t)off,SEEK_SET)==0;
#endif
}
/* what was written is on the drive: a crash after this cannot leave the record without its rows,
 * and the cache's dirty pages stay a few hundred MB (they count against the RAM the plan saw) */
static int k3_sess_commit(FILE *f){
    if(fflush(f)) return 0;
#ifdef _WIN32
    return _commit(_fileno(f))==0;
#else
    return fsync(fileno(f))==0;
#endif
}
static int k3_sess_truncate(FILE *f,int64_t size){
    if(!k3_sess_commit(f)) return 0;
#ifdef _WIN32
    return _chsize_s(_fileno(f),size)==0;
#else
    return ftruncate(fileno(f),(off_t)size)==0;
#endif
}
/* the model, the quantization and the format a file was written by */
static uint64_t k3_sess_fingerprint(Model *m){
    uint64_t h=0xCBF29CE484222325ull;
#define K3_FP(v) do{ uint64_t x_=(uint64_t)(v); for(int b_=0;b_<8;b_++){ h^=(x_>>(8*b_))&255u; h*=0x100000001B3ull; } }while(0)
    const Cfg *c=&m->c;
    K3_FP(K3_SESS_VERSION); K3_FP(c->n_layers); K3_FP(c->hidden); K3_FP(c->vocab); K3_FP(c->kv_lora);
    K3_FP(c->qk_rope); K3_FP(c->index_hd); K3_FP(c->kda_heads); K3_FP(c->kda_hd); K3_FP(c->kda_proj);
    K3_FP(c->conv_k); K3_FP(k3_dsa_indexer_on()); K3_FP(getenv("K3_BITS")?atoi(getenv("K3_BITS")):4);
    K3_FP(getenv("K3_HEAD_BITS")?atoi(getenv("K3_HEAD_BITS")):8);
    for(int i=0;i<c->n_layers;i++){ K3_FP(m->L[i].kda); K3_FP(c->index_hd>0?c->idx_type[i]:0); }
    for(int i=0;i<m->S.nfd;i++){
        struct stat sb;
        K3_FP(m->S.sizes[i]);
        K3_FP(stat(m->S.paths[i],&sb)?0:(int64_t)sb.st_mtime);
    }
#undef K3_FP
    return h;
}
/* A conversation's rows as the files hold them: a position's rows together, every MLA layer's
 * Lc and Rc and, where the indexer is on, its index keys. q NULL: the Model's. */
typedef struct { float *p; int w; } K3RowPart;
static K3RowPart *k3_row_parts(Model *m,const K3Seq *q,int *n_out,size_t *rb_out){
    const Cfg *c=&m->c; const int dsa=c->index_hd>0&&k3_dsa_indexer_on(); int n=0; size_t rb=0;
    K3RowPart *p=malloc(sizeof *p*(size_t)c->n_layers*3);
    if(!p) return NULL;
    for(int i=0;i<c->n_layers;i++) if(!m->L[i].kda){
        float *L=q?(q->Lc?q->Lc[i]:NULL):(m->Lc?m->Lc[i]:NULL);
        float *R=q?(q->Rc?q->Rc[i]:NULL):(m->Rc?m->Rc[i]:NULL);
        float *I=q?(q->Ic?q->Ic[i]:NULL):m->L[i].m.Ic;
        const int keys=dsa&&c->idx_type[i];
        if(!L||!R||(keys&&!I)){ free(p); return NULL; }
        p[n].p=L; p[n++].w=c->kv_lora;
        p[n].p=R; p[n++].w=c->qk_rope;
        rb+=((size_t)c->kv_lora+(size_t)c->qk_rope)*sizeof(float);
        if(keys){ p[n].p=I; p[n++].w=c->index_hd; rb+=(size_t)c->index_hd*sizeof(float); }
    }
    *n_out=n; *rb_out=rb;
    return p;
}
/* positions [a,b) between the rows and the file at its position */
static int k3_rows_io(FILE *f,const K3RowPart *part,int nparts,size_t rb,int a,int b,int write){
    const int step=512;
    uint8_t *buf=malloc((size_t)step*rb);
    if(!buf) return 0;
    int ok=1; size_t since=0;
    for(int t=a;t<b&&ok;t+=step){
        const int n=b-t<step?b-t:step;
        if(!write) ok=fread(buf,rb,(size_t)n,f)==(size_t)n;
        if(ok){
            #pragma omp parallel for schedule(static)
            for(int r=0;r<n;r++){
                uint8_t *row=buf+(size_t)r*rb;
                for(int k=0;k<nparts;k++){
                    const size_t w=(size_t)part[k].w*sizeof(float);
                    float *at=part[k].p+(size_t)(t+r)*part[k].w;
                    if(write) memcpy(row,at,w); else memcpy(at,row,w);
                    row+=w;
                }
            }
        }
        if(write&&ok){
            ok=fwrite(buf,rb,(size_t)n,f)==(size_t)n;
            since+=(size_t)n*rb;
            if(ok&&since>=((size_t)256<<20)){ ok=k3_sess_commit(f); since=0; }
        }
    }
    free(buf);
    return ok;
}
/* every KDA layer's state and windows, in a photo's order */
static int k3_state_io(FILE *f,Model *m,float **ks,float **cq,float **ck,float **cv,int write){
    const Cfg *c=&m->c;
    const size_t s=(size_t)c->kda_heads*c->kda_hd*c->kda_hd, w=(size_t)c->kda_proj*c->conv_k;
    int ok=1;
    for(int i=0;i<c->n_layers&&ok;i++) if(m->L[i].kda){
        float *a[4]={ks[i],cq[i],ck[i],cv[i]}; const size_t n[4]={s,w,w,w};
        for(int j=0;j<4&&ok;j++)
            ok=a[j]&&(write?fwrite(a[j],sizeof(float),n[j],f):fread(a[j],sizeof(float),n[j],f))==n[j];
    }
    return ok;
}

/* ---- the table ---- */
static void k3_sess_free_ram(Model *m,K3Sess *s){
    if(!s->ram) return;
    g_k3_sess_rows-=k3_seq_rows(m,&s->q); g_k3_sess_extra-=k3_sess_extra_of(m,s);
    if(g_k3_sess_rows<0.0) g_k3_sess_rows=0.0;
    if(g_k3_sess_extra<0.0) g_k3_sess_extra=0.0;
    k3_seq_free(m,&s->q);
    free(s->pblob); free(s->plogit);
    s->pblob=NULL; s->plogit=NULL; s->ram=0;
}
static void k3_sess_drop(Model *m,K3Sess *s,int files){
    k3_sess_free_ram(m,s);
    if(files&&s->id) k3_sess_unlink(s->id);
    free(s->fed);
    memset(s,0,sizeof *s);
}
static void k3_sess_unram(Model *m,K3Sess *s,const char *why){
    if(!s->id){
        fprintf(stderr,"[K3-SESS] the %d-token conversation kept aside is dropped (%s): it has no file\n",s->len,why);
        k3_sess_drop(m,s,0);
        return;
    }
    k3_sess_free_ram(m,s);
    if(g_k3_sess_ram>0.0)
        fprintf(stderr,"[K3-SESS] the %d-token conversation kept aside leaves RAM (%s): its file stays\n",s->len,why);
}
/* The kept conversations and the live one's rows past the plan within K3_SESSION_GB, or the kept
 * ones leave RAM, the longest unused first. Called where the caches' size changes
 * (k3_cache_lent_sync): a conversation that grows has the RAM to itself. */
static void k3_sess_yield(Model *m){
    if(g_k3_sess_n<1) return;
    for(;;){
        int v=-1;
        for(int i=0;i<g_k3_sess_n;i++){
            const K3Sess *s=&g_k3_sess[i];
            if(s->on&&s->ram&&(v<0||s->used<g_k3_sess[v].used)) v=i;
        }
        if(v<0) return;
        const double live=g_k3_kv_bytes-g_k3_sess_rows, over=live>g_k3_kv_plan_bytes?live-g_k3_kv_plan_bytes:0.0;
        if(g_k3_sess_rows+g_k3_sess_extra+over<=g_k3_sess_ram) return;
        k3_sess_unram(m,&g_k3_sess[v],g_k3_sess_ram>0.0?"the live conversation and the kept ones are past K3_SESSION_GB":
                      "K3_SESSION_GB=0");
    }
}
/* the files' bytes within K3_SESSION_DISK_GB: the longest unused conversation's go first */
static void k3_sess_disk_trim(Model *m){
    for(;;){
        double tot=g_k3_live_id?g_k3_live_dbytes:0.0; int v=-1;
        for(int i=0;i<g_k3_sess_n;i++){
            const K3Sess *s=&g_k3_sess[i];
            if(!s->on||!s->id) continue;
            tot+=s->dbytes;
            if(v<0||s->used<g_k3_sess[v].used) v=i;
        }
        if(tot<=g_k3_sess_disk||v<0) return;
        K3Sess *s=&g_k3_sess[v];
        fprintf(stderr,"[K3-SESS] the files hold %.1f GB (K3_SESSION_DISK_GB %.0f): the %d-token conversation's, "
                "the longest unused, are deleted\n",tot/1e9,g_k3_sess_disk/1e9,s->len);
        k3_sess_unlink(s->id); s->id=0; s->saved=0; s->dbytes=0.0;
        if(!s->ram) k3_sess_drop(m,s,0);
    }
}
/* a free entry, the longest unused one but `avoid` dropped for it when there is none; -1 when
 * `avoid` is the only one */
static int k3_sess_slot(Model *m,int avoid){
    int v=-1;
    for(int i=0;i<g_k3_sess_n;i++) if(!g_k3_sess[i].on) return i;
    for(int i=0;i<g_k3_sess_n;i++) if(i!=avoid&&(v<0||g_k3_sess[i].used<g_k3_sess[v].used)) v=i;
    if(v<0) return -1;
    fprintf(stderr,"[K3-SESS] %d conversations are kept (K3_SESSIONS): the %d-token one, the longest unused, is dropped\n",
            g_k3_sess_n,g_k3_sess[v].len);
    k3_sess_drop(m,&g_k3_sess[v],1);
    return v;
}
/* A file's record into the table, its state unread: 1 when the file is this model's and whole. */
static int k3_sess_adopt(Model *m,uint64_t id){
    char meta[1024],rows[1024];
    k3_sess_path(meta,sizeof meta,id,"k3s"); k3_sess_path(rows,sizeof rows,id,"k3r");
    FILE *f=fopen(meta,"rb");
    if(!f) return 0;
    K3SessHead h; uint64_t end=0;
    int ok=fread(&h,sizeof h,1,f)==1&&h.magic==K3_SESS_MAGIC&&h.fp==g_k3_sess_fp&&h.id==id&&h.len>=1&&h.len<(1<<26)&&
           h.rowbytes==(int32_t)g_k3_kv_row_bytes&&h.state_floats==(uint64_t)k3_ckpt_blob_floats(m)&&
           h.vocab==m->c.vocab&&h.ppos>=0&&h.ppos<h.len;
    int *fed=ok?malloc((size_t)h.len*sizeof(int)):NULL;
    ok=ok&&fed&&fread(fed,sizeof(int),(size_t)h.len,f)==(size_t)h.len;
    const int64_t state=ok?(int64_t)h.state_floats*(int64_t)sizeof(float):0;
    const int64_t expect=ok?(int64_t)sizeof h+(int64_t)h.len*(int64_t)sizeof(int)+state*(h.ppos>0?2:1)+
                         (h.ppos>0&&h.plog?(int64_t)h.vocab*(int64_t)sizeof(float):0)+8:0;
    ok=ok&&k3_sess_seek(f,expect-8)&&fread(&end,sizeof end,1,f)==1&&end==K3_SESS_END;
    fclose(f);
    const int64_t msize=ok?k3_sess_fsize(meta):-1, rsize=ok?k3_sess_fsize(rows):-1;
    ok=ok&&msize==expect&&rsize==(int64_t)h.len*h.rowbytes;
    const int si=ok?k3_sess_slot(m,-1):-1;
    if(si<0){ free(fed); return 0; }
    K3Sess *s=&g_k3_sess[si];
    s->on=1; s->ram=0; s->id=id; s->saved=h.len; s->dbytes=(double)msize+(double)rsize;
    s->fed=fed; s->len=h.len; s->ppos=h.ppos; s->plog=h.ppos>0&&h.plog; s->used=h.stamp;
    return 1;
}
/* The directory at the start: this model's whole files into the table, the newest while it has
 * room; the rest of what this code writes there is deleted. */
static void k3_sess_scan(Model *m){
    DIR *d=opendir(g_k3_sess_dir);
    if(!d) return;
    uint64_t *ids=NULL; int n=0,cap=0; struct dirent *e;
    while((e=readdir(d))){
        char *endp=NULL; const unsigned long long id=strtoull(e->d_name,&endp,10);
        if(endp!=e->d_name+8||!id||(strcmp(endp,".k3s")&&strcmp(endp,".k3r")&&strcmp(endp,".k3s.tmp"))) continue;
        int seen=0;
        for(int i=0;i<n;i++) if(ids[i]==id) seen=1;
        if(seen) continue;
        if(n==cap){ cap=cap?cap*2:64; ids=realloc(ids,(size_t)cap*sizeof *ids); if(!ids){ closedir(d); return; } }
        ids[n++]=id;
    }
    closedir(d);
    for(int i=1;i<n;i++){                       /* the newest numbers first */
        const uint64_t v=ids[i]; int k=i;
        for(;k>0&&ids[k-1]<v;k--) ids[k]=ids[k-1];
        ids[k]=v;
    }
    for(int i=0;i<n;i++){
        if(ids[i]>=g_k3_sess_next) g_k3_sess_next=ids[i]+1;
        int full=1;
        for(int k=0;k<g_k3_sess_n;k++) if(!g_k3_sess[k].on) full=0;
        if(full||!k3_sess_adopt(m,ids[i])){
            fprintf(stderr,"[K3-SESS] %s/%08llu.*: %s, deleted\n",g_k3_sess_dir,(unsigned long long)ids[i],
                    full?"more conversations than K3_SESSIONS":"not a whole conversation of this model");
            k3_sess_unlink(ids[i]);
        }
    }
    free(ids);
}

/* ---- the photo kept with a conversation ---- */
/* out of the pool: the deepest photo below the conversation's end (its last prompt's end) */
static void k3_sess_take_photo(K3Sess *s){
    s->ppos=0; s->plog=0; s->pblob=NULL; s->plogit=NULL;
    if(g_k3_ckpt_dir) return;                     /* the photos are files of their own there */
    int best=-1;
    for(int i=0;i<g_k3_ckpt_slots;i++){
        const K3Ckpt *k=&g_k3_ckpt[i];
        if(!k->blob||k->pinned||k->pos<1||k->pos>=s->len) continue;
        if(memcmp(k->fed,s->fed,(size_t)k->pos*sizeof(int))) continue;
        if(best<0||k->pos>g_k3_ckpt[best].pos) best=i;
    }
    if(best<0) return;
    K3Ckpt *k=&g_k3_ckpt[best];
    s->ppos=k->pos; s->pblob=k->blob; s->plogit=k->logit; s->plog=k->logit!=NULL;
    free(k->fed);
    memset(k,0,sizeof *k);
}
/* back into the pool, where k3_ckpt_restore_best finds it */
static void k3_sess_give_photo(K3Sess *s){
    if(!s->pblob){ free(s->plogit); s->plogit=NULL; return; }
    int v=-1;
    for(int pass=0;pass<2&&v<0;pass++)
        for(int i=0;i<g_k3_ckpt_slots;i++){
            const K3Ckpt *k=&g_k3_ckpt[i];
            if(!pass&&k->pinned) continue;
            if(!k->blob){ v=i; break; }
            if(v<0||k->used<g_k3_ckpt[v].used) v=i;
        }
    int *fed=v>=0&&!g_k3_ckpt_dir?malloc((size_t)s->ppos*sizeof(int)):NULL;
    if(!fed){ free(s->pblob); free(s->plogit); }
    else {
        K3Ckpt *k=&g_k3_ckpt[v];
        free(k->blob); free(k->fed); free(k->logit);
        memcpy(fed,s->fed,(size_t)s->ppos*sizeof(int));
        k->blob=s->pblob; k->fed=fed; k->logit=s->plogit; k->pos=s->ppos; k->pinned=0; k->used=++g_k3_ckpt_clock;
    }
    s->pblob=NULL; s->plogit=NULL;
}
/* The deepest photo of the pool that serves ids[0..np) on rows holding its first `common`
 * tokens (k3_ckpt_restore_best's rule): its position, 0 when there is none. */
static int k3_sess_pool_gain(const int *ids,int np,int common,int *slot){
    const int whole=common>=np, live=whole?np:common; int best=-1;
    for(int s=0;s<g_k3_ckpt_slots;s++){
        const K3Ckpt *k=&g_k3_ckpt[s];
        if(!k3_ckpt_slot_full(k)||k->pos<1||k->pos>live) continue;
        if(k->pos>=np&&!(whole&&k->pos==np&&k->logit)) continue;
        if(memcmp(k->fed,ids,(size_t)k->pos*sizeof(int))) continue;
        if(best<0||k->pos>g_k3_ckpt[best].pos) best=s;
    }
    if(slot) *slot=best;
    return best<0?0:g_k3_ckpt[best].pos;
}
/* What a conversation (its tokens fed[0..len), the photo kept with it at ppos) saves of the
 * prompt. *same: the prompt is that conversation's -- it continues it, or parts from it past
 * `key` -- and then its rows, the photo kept with it or one of the pool's serve it; else only
 * one of the pool's photos below where they part, whose rows it lends. */
static int k3_sess_gain(const int *fed,int len,int ppos,int plog,const int *ids,int np,int key,int *same,int *common_out){
    const int common=k3_common(fed,len,ids,np), cont=common==len&&len<np;
    *common_out=common; *same=cont||common>=key;
    if(cont) return len;
    int g=k3_sess_pool_gain(ids,np,common,NULL);
    if(*same&&ppos>g&&ppos<=common&&(ppos<np||(ppos==np&&plog))) g=ppos;
    return g;
}

/* ---- the live conversation's file ---- */
static int k3_live_has(const Model *m){ return m->kvp.fed&&!m->kvp.tainted&&m->kvp.len>0; }
static int k3_live_keepable(const Model *m){ return k3_live_has(m)&&m->Lc&&m->kvp.len>=g_k3_sess_min; }
/* The live conversation to its files: the rows past those the file already holds, then its
 * tokens, the KDA state and the photo below its end. The old record goes first -- the rows it
 * describes are about to change -- and the new one is renamed into place last. */
static int k3_live_save(Model *m,const char *why){
    if(!g_k3_sess_dir[0]||!k3_live_keepable(m)) return 0;
    const int len=m->kvp.len;
    if(g_k3_live_id&&!g_k3_live_dirty&&g_k3_live_saved>=len) return 1;
    int nparts=0; size_t rb=0;
    const double t0=now_s(), state=k3_state_bytes(m);
#ifdef COLI_VULKAN
    k3c_sync_host(m);
#endif
    K3RowPart *part=k3_row_parts(m,NULL,&nparts,&rb);
    if(!part||rb!=(size_t)g_k3_kv_row_bytes){ free(part); return 0; }
    if((double)len*(double)rb+2.0*state>g_k3_sess_disk){
        fprintf(stderr,"[K3-SESS] the %d-token conversation is larger than K3_SESSION_DISK_GB (%.0f): not written\n",
                len,g_k3_sess_disk/1e9);
        free(part);
        if(g_k3_live_id) k3_sess_unlink(g_k3_live_id);
        g_k3_live_id=0; g_k3_live_saved=0; g_k3_live_dbytes=0.0;
        return 0;
    }
    if(!g_k3_live_id){ g_k3_live_id=g_k3_sess_next++; g_k3_live_saved=0; }
    char meta[1024],tmp[1040],rows[1024];
    k3_sess_path(meta,sizeof meta,g_k3_live_id,"k3s"); k3_sess_path(tmp,sizeof tmp,g_k3_live_id,"k3s.tmp");
    k3_sess_path(rows,sizeof rows,g_k3_live_id,"k3r");
    remove(meta);
    int from=g_k3_live_saved<len?g_k3_live_saved:len;
    if(from>0&&k3_sess_fsize(rows)<(int64_t)from*(int64_t)rb) from=0;   /* not the file these rows went to */
    FILE *fr=from>0?fopen(rows,"r+b"):NULL;
    if(!fr){ from=0; fr=fopen(rows,"wb"); }
    int ok=fr!=NULL;
    if(ok){
        setvbuf(fr,NULL,_IOFBF,(size_t)8<<20);
        ok=k3_sess_seek(fr,(int64_t)from*(int64_t)rb)&&k3_rows_io(fr,part,nparts,rb,from,len,1)&&
           k3_sess_truncate(fr,(int64_t)len*(int64_t)rb);
        ok=fclose(fr)==0&&ok;
    }
    free(part);
    /* the photo below the conversation's end, as it stands in the pool */
    const K3Ckpt *ph=NULL;
    for(int i=0;ok&&!g_k3_ckpt_dir&&i<g_k3_ckpt_slots;i++){
        const K3Ckpt *k=&g_k3_ckpt[i];
        if(!k->blob||k->pinned||k->pos<1||k->pos>=len||memcmp(k->fed,m->kvp.fed,(size_t)k->pos*sizeof(int))) continue;
        if(!ph||k->pos>ph->pos) ph=k;
    }
    K3SessHead h; memset(&h,0,sizeof h);
    h.magic=K3_SESS_MAGIC; h.fp=g_k3_sess_fp; h.id=g_k3_live_id; h.stamp=++g_k3_sess_clock;
    h.state_floats=(uint64_t)k3_ckpt_blob_floats(m); h.len=len; h.ppos=ph?ph->pos:0; h.plog=ph&&ph->logit;
    h.vocab=m->c.vocab; h.rowbytes=(int32_t)rb;
    FILE *fm=ok?fopen(tmp,"wb"):NULL;
    ok=fm!=NULL;
    if(ok){
        const uint64_t end=K3_SESS_END;
        setvbuf(fm,NULL,_IOFBF,(size_t)8<<20);
        ok=fwrite(&h,sizeof h,1,fm)==1&&fwrite(m->kvp.fed,sizeof(int),(size_t)len,fm)==(size_t)len&&
           k3_state_io(fm,m,m->kstate,m->cwq,m->cwk,m->cwv,1);
        if(ok&&ph) ok=fwrite(ph->blob,sizeof(float),(size_t)h.state_floats,fm)==(size_t)h.state_floats&&
                      (!ph->logit||fwrite(ph->logit,sizeof(float),(size_t)h.vocab,fm)==(size_t)h.vocab);
        ok=ok&&fwrite(&end,sizeof end,1,fm)==1&&k3_sess_commit(fm);
        ok=fclose(fm)==0&&ok;
        ok=ok&&rename(tmp,meta)==0;
    }
    if(!ok){
        fprintf(stderr,"[K3-SESS] writing the conversation to %s failed (%s): it is not kept in a file\n",
                g_k3_sess_dir,strerror(errno));
        k3_sess_unlink(g_k3_live_id);
        g_k3_live_id=0; g_k3_live_saved=0; g_k3_live_dbytes=0.0;
        return 0;
    }
    const double head=(double)sizeof h+(double)len*sizeof(int)+state*(ph?2.0:1.0)+(h.plog?(double)h.vocab*sizeof(float):0.0)+8.0;
    const double wrote=(double)(len-from)*(double)rb+head;
    g_k3_live_saved=len; g_k3_live_dirty=0;
    g_k3_live_dbytes=(double)len*(double)rb+head;
    fprintf(stderr,"[K3-SESS] the %d-token conversation written to its file (%s): %.2f GB in %.1fs, its rows from "
            "position %d, the state at its end%s\n",len,why,wrote/1e9,now_s()-t0,from,
            ph?" and the photo of its last prompt's end":"");
    k3_sess_disk_trim(m);
    return 1;
}
/* A kept conversation's state from its file into s->q. */
static int k3_sess_load(Model *m,K3Sess *s){
    const Cfg *c=&m->c;
    char meta[1024],rows[1024];
    k3_sess_path(meta,sizeof meta,s->id,"k3s"); k3_sess_path(rows,sizeof rows,s->id,"k3r");
    K3Seq *q=&s->q;
    FILE *fm=fopen(meta,"rb"), *fr=fopen(rows,"rb");
    K3SessHead h; uint64_t end=0;
    int ok=fm&&fr&&fread(&h,sizeof h,1,fm)==1&&h.magic==K3_SESS_MAGIC&&h.fp==g_k3_sess_fp&&h.id==s->id&&
           h.len==s->len&&h.rowbytes==(int32_t)g_k3_kv_row_bytes&&h.state_floats==(uint64_t)k3_ckpt_blob_floats(m)&&
           h.vocab==c->vocab;
    int *fed=ok?malloc((size_t)s->len*sizeof(int)):NULL;
    ok=ok&&fed&&fread(fed,sizeof(int),(size_t)s->len,fm)==(size_t)s->len&&!memcmp(fed,s->fed,(size_t)s->len*sizeof(int));
    free(fed);
    if(!ok){ if(fm) fclose(fm); if(fr) fclose(fr); return 0; }
    k3_mem_make_room(m,(double)s->len*g_k3_kv_row_bytes+2.0*k3_state_bytes(m),"a kept conversation read from its file");
    k3_seq_alloc(m,q);
    q->Lc=calloc(c->n_layers,sizeof(float*)); q->Rc=calloc(c->n_layers,sizeof(float*));
    if(!q->Lc||!q->Rc){ fprintf(stderr,"OOM a kept conversation's rows\n"); exit(1); }
    const int dsa=c->index_hd>0&&k3_dsa_indexer_on();
    for(int i=0;i<c->n_layers;i++) if(!m->L[i].kda){
        q->Lc[i]=falloc((int64_t)s->len*c->kv_lora);
        q->Rc[i]=falloc((int64_t)s->len*c->qk_rope);
        if(dsa&&c->idx_type[i]) q->Ic[i]=falloc((int64_t)s->len*c->index_hd);
    }
    q->kv_cap=s->len; q->max_t=s->len;
    g_k3_kv_bytes+=k3_seq_rows(m,q);              /* k3_seq_free takes them back */
    int nparts=0; size_t rb=0;
    K3RowPart *part=k3_row_parts(m,q,&nparts,&rb);
    ok=part&&rb==(size_t)h.rowbytes&&kv_prefix_alloc(&q->kvp,s->len);
    if(ok){
        setvbuf(fr,NULL,_IOFBF,(size_t)8<<20);
        memcpy(q->kvp.fed,s->fed,(size_t)s->len*sizeof(int)); q->kvp.len=s->len;
        ok=k3_rows_io(fr,part,nparts,rb,0,s->len,0)&&k3_state_io(fm,m,q->kstate,q->cwq,q->cwk,q->cwv,0);
    }
    free(part);
    s->pblob=NULL; s->plogit=NULL;
    if(ok&&h.ppos>0&&!g_k3_ckpt_dir){
        s->pblob=malloc((size_t)h.state_floats*sizeof(float));
        ok=s->pblob&&fread(s->pblob,sizeof(float),(size_t)h.state_floats,fm)==(size_t)h.state_floats;
        if(ok&&h.plog){
            s->plogit=malloc((size_t)h.vocab*sizeof(float));
            ok=s->plogit&&fread(s->plogit,sizeof(float),(size_t)h.vocab,fm)==(size_t)h.vocab;
        }
        ok=ok&&fread(&end,sizeof end,1,fm)==1&&end==K3_SESS_END;
    }
    fclose(fm); fclose(fr);
    if(!ok){
        k3_seq_free(m,q); free(s->pblob); free(s->plogit); s->pblob=NULL; s->plogit=NULL;
        return 0;
    }
    s->ppos=s->pblob?h.ppos:0; s->plog=s->plogit!=NULL; s->ram=1;
    g_k3_sess_rows+=k3_seq_rows(m,q); g_k3_sess_extra+=k3_sess_extra_of(m,s);
    return 1;
}

/* ---- a request's conversation ---- */
/* A conversation that left the Model into the table, with its file and the photo below its end:
 * the entry, or -1 when there was no room (then it is dropped). */
static int k3_sess_keep(Model *m,K3Seq *q,uint64_t id,int saved,double dbytes,const char *why,int avoid){
    const int si=k3_sess_slot(m,avoid);
    int *fed=si>=0?malloc((size_t)q->kvp.len*sizeof(int)):NULL;
    if(!fed){
        g_k3_sess_rows-=k3_seq_rows(m,q); k3_seq_free(m,q);
        if(id) k3_sess_unlink(id);
        return -1;
    }
    memcpy(fed,q->kvp.fed,(size_t)q->kvp.len*sizeof(int));
    K3Sess *s=&g_k3_sess[si];
    s->on=1; s->ram=1; s->id=id; s->saved=saved; s->dbytes=dbytes; s->fed=fed; s->len=q->kvp.len; s->q=*q;
    s->used=++g_k3_sess_clock;
    memset(q,0,sizeof *q);
    k3_sess_take_photo(s);
    g_k3_sess_extra+=k3_sess_extra_of(m,s);
    fprintf(stderr,"[K3-SESS] the %d-token conversation kept aside (%s)%s%s\n",s->len,why,
            s->id?(g_k3_sess_ram>0.0?", in RAM and in its file":", in its file"):", in RAM",
            s->pblob?", with the photo of its last prompt's end":"");
    return si;
}
/* The live conversation aside -- written first where there are files, then out of the Model
 * whole -- or dropped when it is too short to keep. The Model then holds an empty state: no
 * rows, the KDA state zero. The entry it went to, or -1. */
static int k3_sess_live_aside(Model *m,const char *why,int avoid){
    int si=-1;
    if(!k3_live_keepable(m)){
        if(g_k3_live_id) k3_sess_unlink(g_k3_live_id);
        model_state_reset(m);
    } else {
        k3_live_save(m,why);
        const int len=m->kvp.len;
        if(m->kv_cap>len+len/4) kv_rows_to(m,len);   /* the rows written, not the room they doubled into */
#ifdef COLI_VULKAN
        k3c_sync_host(m);            /* the dense chain may hold the newest recurrent state */
#endif
        K3Seq old;
        k3_seq_alloc(m,&old);
        k3_seq_swap(m,&old);
        g_k3_sess_rows+=k3_seq_rows(m,&old);
#ifdef COLI_VULKAN
        k3c_host_wrote(m,1);
#endif
        si=k3_sess_keep(m,&old,g_k3_live_id,g_k3_live_saved,g_k3_live_dbytes,why,avoid);
    }
    g_k3_live_id=0; g_k3_live_saved=0; g_k3_live_dbytes=0.0; g_k3_live_dirty=0;
    return si;
}
/* a kept conversation becomes the live one */
static int k3_sess_enter(Model *m,int si){
    K3Sess s=g_k3_sess[si];
    memset(&g_k3_sess[si],0,sizeof(K3Sess));
    const double t0=now_s(); const int from_file=!s.ram;
    k3_sess_live_aside(m,"a kept conversation comes back",-1);
    k3_cache_lent_sync(m);                        /* it leaves RAM before this one is read, if it is to */
    if(!s.ram&&!k3_sess_load(m,&s)){
        fprintf(stderr,"[K3-SESS] the %d-token conversation's file cannot be read: dropped\n",s.len);
        k3_sess_unlink(s.id); free(s.fed);
        return 0;
    }
    g_k3_sess_rows-=k3_seq_rows(m,&s.q); g_k3_sess_extra-=k3_sess_extra_of(m,&s);
    k3_seq_swap(m,&s.q);          /* the Model holds it; s.q the empty state that was there */
    k3_seq_free(m,&s.q);
    k3_sess_give_photo(&s);
    g_k3_live_id=s.id; g_k3_live_saved=s.saved; g_k3_live_dbytes=s.dbytes; g_k3_live_dirty=0;
    free(s.fed);
#ifdef COLI_VULKAN
    k3c_host_rewrote(m);
#endif
    k3_cache_lent_sync(m);
    if(from_file) fprintf(stderr,"[K3-SESS] the %d-token conversation is back, read from its file in %.1fs\n",s.len,now_s()-t0);
    else fprintf(stderr,"[K3-SESS] the %d-token conversation is back from RAM\n",s.len);
    return 1;
}
/* The new conversation, in the Model's empty state, begins with the P tokens it shares with the
 * kept conversation si, up to the pool's photo `slot`: those rows copied (from RAM or from the
 * file), the KDA state the photo's. 0 when they cannot be had: the state is empty again. */
static int k3_sess_prefix(Model *m,int si,int P,int slot,const int *ids,int np,int max_t){
    const Cfg *c=&m->c; const int dsa=c->index_hd>0&&k3_dsa_indexer_on();
    const double t0=now_s();
    K3Sess *s=si>=0?&g_k3_sess[si]:NULL;
    int ok=s&&s->on&&s->len>=P&&P>0&&slot>=0&&(s->ram||(s->id&&s->saved>=P));
    if(ok){ kv_alloc_cap(m,max_t,k3_kv_grow_on()?np+k3_kv_gen_rows():max_t); ok=m->Lc&&m->kvp.fed&&m->kv_cap>=P; }
    if(ok&&s->ram){
        for(int i=0;ok&&i<c->n_layers;i++) if(!m->L[i].kda){
            const int keys=dsa&&c->idx_type[i];
            ok=m->Lc[i]&&m->Rc[i]&&s->q.Lc&&s->q.Lc[i]&&s->q.Rc&&s->q.Rc[i]&&(!keys||(m->L[i].m.Ic&&s->q.Ic&&s->q.Ic[i]));
            if(!ok) break;
            memcpy(m->Lc[i],s->q.Lc[i],(size_t)P*c->kv_lora*sizeof(float));
            memcpy(m->Rc[i],s->q.Rc[i],(size_t)P*c->qk_rope*sizeof(float));
            if(keys) memcpy(m->L[i].m.Ic,s->q.Ic[i],(size_t)P*c->index_hd*sizeof(float));
        }
    } else if(ok){
        char rows[1024]; int nparts=0; size_t rb=0;
        k3_sess_path(rows,sizeof rows,s->id,"k3r");
        FILE *fr=fopen(rows,"rb");
        K3RowPart *part=k3_row_parts(m,NULL,&nparts,&rb);
        ok=fr&&part&&rb==(size_t)g_k3_kv_row_bytes;
        if(ok){ setvbuf(fr,NULL,_IOFBF,(size_t)8<<20); ok=k3_rows_io(fr,part,nparts,rb,0,P,0); }
        free(part);
        if(fr) fclose(fr);
    }
    K3Ckpt *k=ok?&g_k3_ckpt[slot]:NULL;
    if(ok){
        if(g_k3_ckpt_dir) ok=k3_ckpt_disk_read(m,slot,k->pos);
        else k3_ckpt_copy(k->blob,m,0);
    }
    if(!ok){ model_state_reset(m); return 0; }
    kv_prefix_record(&m->kvp,ids,0,P);
    k->used=++g_k3_ckpt_clock; s->used=++g_k3_sess_clock;
#ifdef COLI_VULKAN
    k3c_host_rewrote(m);
#endif
    fprintf(stderr,"[K3-SESS] the new conversation starts after the %d tokens it shares with a kept one (their rows "
            "copied from %s in %.2fs)\n",P,s->ram?"RAM":"its file",now_s()-t0);
    return 1;
}
/* Before a request's state is prepared: the conversation that saves most of its prompt becomes
 * the live one, and the prompt of another conversation puts the live one aside. `key`: the
 * tokens up to the end of the prompt's first record that is not the system's (np for a raw
 * prompt). */
static void k3_sess_pick(Model *m,const int *ids,int np,int max_t,int key){
    if(!k3_sess_on()) return;
    for(int tries=0;tries<=K3_SESS_MAX;tries++){
        int lsame=0,lcommon=0,best=-1,bsame=0,bcommon=0;
        const int lhas=k3_live_has(m);
        int bgain=lhas?k3_sess_gain(m->kvp.fed,m->kvp.len,0,0,ids,np,key,&lsame,&lcommon):0;
        for(int i=0;i<g_k3_sess_n;i++){
            const K3Sess *s=&g_k3_sess[i];
            if(!s->on) continue;
            int sm=0,cm=0;
            const int g=k3_sess_gain(s->fed,s->len,s->ppos,s->plog,ids,np,key,&sm,&cm);
            if(g>bgain||(best>=0&&g==bgain&&s->ram&&!g_k3_sess[best].ram)){ best=i; bgain=g; bsame=sm; bcommon=cm; }
        }
        if(best>=0&&bsame){                       /* a kept conversation's own turn */
            if(k3_sess_enter(m,best)) return;
            continue;                             /* its file is gone: the best of the rest */
        }
        if(best<0&&(!lhas||lsame)) return;        /* the live conversation's own turn, or nothing to keep */
        if(best<0&&!k3_live_keepable(m)) return;  /* too short to keep: the new one writes over it, as without this */
        /* another conversation: the live one aside, the new one after what it shares with a kept one */
        int slot=-1;
        const int P=best>=0?k3_sess_pool_gain(ids,np,bcommon,&slot):k3_sess_pool_gain(ids,np,lcommon,&slot);
        if(best>=0) g_k3_sess[best].used=++g_k3_sess_clock;
        const int kept=k3_sess_live_aside(m,"another conversation begins",best);
        const int donor=best>=0?best:kept;
        if(P>0&&donor>=0) k3_sess_prefix(m,donor,P,slot,ids,np,max_t);
        k3_cache_lent_sync(m);
        return;
    }
}
/* a cancelled prefill dropped the live state: its file, if it has one, is a kept conversation again */
static void k3_sess_live_lost(Model *m){
    if(k3_sess_on()&&g_k3_live_id&&!k3_sess_adopt(m,g_k3_live_id)) k3_sess_unlink(g_k3_live_id);
    g_k3_live_id=0; g_k3_live_saved=0; g_k3_live_dbytes=0.0; g_k3_live_dirty=0;
}
/* 1: a command, or the end of the input, is there to read; 0: nothing yet; -1: stdin cannot be
 * polled (a console, a file). coli_stdin_readable says 0 for a pipe whose writer has closed it:
 * PeekNamedPipe fails with ERROR_BROKEN_PIPE there, and the wait below sat out its seconds while
 * the gateway, which closes stdin to stop the engine and kills it 30 s later, cut the write. */
static int k3_sess_stdin_state(void){
#ifdef _WIN32
    HANDLE ih=(HANDLE)_get_osfhandle(_fileno(stdin)); DWORD avail=0;
    if(ih==INVALID_HANDLE_VALUE) return -1;
    if(PeekNamedPipe(ih,NULL,0,NULL,&avail,NULL)) return avail>0;
    return GetLastError()==ERROR_BROKEN_PIPE?1:-1;
#else
    return coli_stdin_readable();
#endif
}
/* Before the serve waits for a command: no request for K3_SESSION_IDLE seconds and the live
 * conversation is written, so a server stopped without a word loses little. The end of the
 * input returns at once: the exit writes it (k3_sess_exit). */
static void k3_sess_idle(Model *m){
    if(!k3_sess_on()||!g_k3_sess_dir[0]||g_k3_sess_idle<=0||!g_k3_live_dirty||g_k3_park.on||!k3_live_keepable(m)) return;
    const double t0=now_s();
    while(k3_sess_stdin_state()==0){
        if(now_s()-t0>=(double)g_k3_sess_idle){ k3_live_save(m,"the server idle"); return; }
        usleep(50000);
    }
}
static void k3_sess_exit(Model *m){
    if(k3_sess_on()&&g_k3_sess_dir[0]&&!g_k3_park.on) k3_live_save(m,"the server stops");
}
static void k3_sess_init(Model *m){
    const char *e=getenv("K3_SESSIONS");
    int n=e?atoi(e):0;
    if(n>K3_SESS_MAX) n=K3_SESS_MAX;
    if(n<1) return;
    if(g_k3_mux_slots!=1||g_k3_metal){
        fprintf(stderr,"[K3-SESS] K3_SESSIONS is off %s\n",g_k3_metal?"under Metal":"in a multiplexed serve (KV_SLOTS > 1)");
        return;
    }
    g_k3_sess_n=n;
    if((e=getenv("K3_SESSION_DIR"))&&*e){
        if(strlen(e)>=sizeof g_k3_sess_dir-32) fprintf(stderr,"[K3-SESS] K3_SESSION_DIR is too long: no files\n");
        else snprintf(g_k3_sess_dir,sizeof g_k3_sess_dir,"%s",e);
    }
    g_k3_sess_ram=g_k3_sess_dir[0]?0.0:8e9;       /* with files, kept conversations take no RAM from the experts */
    if((e=getenv("K3_SESSION_GB"))&&*e) g_k3_sess_ram=atof(e)*1e9;
    if(g_k3_sess_ram<0.0) g_k3_sess_ram=0.0;
    if((e=getenv("K3_SESSION_DISK_GB"))&&*e) g_k3_sess_disk=atof(e)*1e9;
    if((e=getenv("K3_SESSION_MIN"))&&*e){ g_k3_sess_min=atoi(e); if(g_k3_sess_min<2) g_k3_sess_min=2; }
    if((e=getenv("K3_SESSION_IDLE"))&&*e) g_k3_sess_idle=atoi(e);
    g_k3_sess_clock=(uint64_t)time(NULL);
    g_k3_sess_fp=k3_sess_fingerprint(m);
    int found=0; double gb=0.0;
    if(g_k3_sess_dir[0]){
#ifdef _WIN32
        _mkdir(g_k3_sess_dir);
#else
        mkdir(g_k3_sess_dir,0755);
#endif
        k3_sess_scan(m);
        for(int i=0;i<g_k3_sess_n;i++) if(g_k3_sess[i].on){ found++; gb+=g_k3_sess[i].dbytes/1e9; }
    }
    fprintf(stderr,"[K3-SESS] up to %d conversations kept beside the live one (K3_SESSIONS) from %d tokens, ",n,g_k3_sess_min);
    if(g_k3_sess_ram>0.0)
        fprintf(stderr,"%.1f GB of RAM for them and the live one's rows past the plan together (K3_SESSION_GB: a "
                "long conversation has the RAM to itself)",g_k3_sess_ram/1e9);
    else fprintf(stderr,"none in RAM (K3_SESSION_GB=0: the expert cache is whole)");
    if(g_k3_sess_dir[0]) fprintf(stderr,"; files in %s, up to %.0f GB: %d conversations found there (%.1f GB); the "
            "live one written after %d s idle and at exit\n",g_k3_sess_dir,g_k3_sess_disk/1e9,found,gb,g_k3_sess_idle);
    else fprintf(stderr,"; no files (K3_SESSION_DIR): a conversation that leaves RAM is gone\n");
}

#endif /* K3_SESSIONS_H */
