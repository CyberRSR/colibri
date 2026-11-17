/* k3_draft.h -- a draft model beside Kimi K3.1 Pro (K3_DRAFT=<its folder>; included by kimi_k3.c).
 *
 * A decode step costs what one token's 93 layers cost, and part of that is the same whether
 * the step carries one row or several: the dense matrices stream once a step. So when the next
 * tokens can be guessed, the step that feeds the last token also feeds the guesses behind it
 * -- a block -- and every guess the model itself then picks was a token it did not need a step
 * of its own for. The guesses come from Inferact/Kimi-K3.1-Pro-DSpark, a draft model trained
 * for this target: five small layers that read the target's hidden rows after layers 75, 79,
 * 83, 87 and 91 (its aux_hidden_state_layer_ids 76 .. 92, counted from 1) and propose 7 tokens
 * in one pass (backend_cuda_k3_draft.cuh; the device only).
 *
 * What the reply is does not change. Every token is still sampled from the target's own
 * logits, in order, with the sampler the plain loop uses: a draft only decides whether the row
 * that follows was computed in advance. A draft the model does not pick ends the block, and
 * everything the rows past it wrote is taken back:
 *   - the KDA layers' recurrent state. A block's first row -- the token already chosen --
 *     updates it in place as any step does; row t >= 1 writes its state beside it
 *     (k3_kda_head: g_k3d_chain[t-1]), and the row the block ends on becomes the live state by
 *     a swap of pointers. No state is copied and none is recomputed.
 *   - the conv windows, rebuilt from their copy before the block and the block's raw
 *     projections (k3d_kda_save): the last K of the two in order.
 *   - the MLA caches and K3.1's index keys are rows by position: the rows past the block's end
 *     are written again when those positions come. The record of the tokens fed is cut back.
 * A row is computed alone in a step of any size, so a block's rows are the rows the same
 * tokens would get one step at a time: K3_LOGITS with K3_LOGITS_DECODE=1 writes the same file
 * with the draft model as without it (tests/kimi_k3_draft_harness.py does that with drafts
 * read from a file, right ones and wrong ones, on any build).
 *
 * The draft model's context is a ring of the last positions' keys and values on the device.
 * A step's tapped rows go there as it ends (k3d_step_end); a request that restores an earlier
 * state keeps the ring's rows up to that position when they were computed from the same
 * tokens (a hash of the prefix at every row, k3d_sync) and starts it empty otherwise.
 *
 * How many drafts a block carries is a wager: a row costs about 0.7 of a step here (its
 * routed experts are its own, read and computed for it), so a draft pays only when it is very
 * likely to stand. The draft model's own probability of its pick decides: the leading drafts
 * at K3_DRAFT_P or more, at most K3_DRAFT_MAX.
 *
 * K3_DRAFT=<folder>     the draft model (config.json, model.safetensors); unset: none of this runs
 * K3_DRAFT_P=0.98       a draft is checked when its probability is at least this
 * K3_DRAFT_MAX=3        the most drafts a block carries (1..7); each is 0.43 GB of state beside the live one
 * K3_DRAFT_MINCTX=256   the fewest context rows the ring must hold (fewer when the conversation is shorter)
 * K3_DRAFT_LOG=1        a line a block
 * K3_DRAFT_DUMP=prefix  (tests) the context rows as the device gets them (<prefix>.taps.f32) and every
 *                       pass's 7 picks and their probabilities (<prefix>.events.txt, in order), for a
 *                       reference implementation to compute the same from
 * K3_DRAFT_ORACLE=file  (tests) the drafts from a file instead of a draft model, lines of "position id":
 *                       the guess for the token at that position. Any build, no device: the blocks and
 *                       what they take back, with guesses a test chose */
#ifndef K3_DRAFT_H
#define K3_DRAFT_H

#define K3D_MAXN 7
typedef struct {
    int blocks;                                  /* a step may carry drafts: the draft model's (on) or K3_DRAFT_ORACLE's */
    int on, loaded;                              /* the device holds the draft model; its weights are there, the head not yet */
    int D, T, L, B, win, ring, R, V, mask_id;
    int tap_layer[8]; int8_t tap_of[256];        /* the tapped layers, 0-based; layer -> its place + 1 */
    uint16_t *w1;                                /* the Markov embedding, bf16 [V][R]: a row a pick */
    float *e0, *me;                              /* a draft's: the last token's embedding, a Markov row */
    /* a step's tapped rows: rows tap_r0 .. C-1 of the running step, [row][T D] */
    float *taps; int64_t tap_cap; int tap_r0, tap_C;
    /* the ring: positions [lo, hi) are on the device, hash[position mod ring] the hash of the tokens through it */
    int lo, hi; uint64_t *hash, ph; int ph_pos;
    int *orc; int orc_n;                         /* K3_DRAFT_ORACLE: the guess for the token at a position, -1 none */
    /* the block being checked: ids[0] the token fed, ids[1..n] the drafts; rows 0..j taken so far */
    int n, j, pos0, ids[1+K3D_MAXN];
    float *mix;                                  /* [1+K3D_MAXN][D]: what the head reads of each row */
    float **chain[K3D_MAXN];                     /* [t-1][layer]: a KDA layer's state after the block's row t */
    float **win0, **raw;                         /* [layer]: its three windows before the block, the block's raw q, k, v */
    int max_n, minctx, log; float min_p;
    /* a request's counts */
    uint64_t s_steps, s_blocks, s_drafts, s_kept, s_passes; double s_time;
    uint64_t s_at[K3D_MAXN], s_at_ok[K3D_MAXN];
    FILE *dump_ev, *dump_taps;                   /* K3_DRAFT_DUMP */
} K3Draft;
static K3Draft g_k3d;

static int k3d_wanted(void){
    const char *a=getenv("K3_DRAFT"), *b=getenv("K3_DRAFT_ORACLE");
    return (a&&*a)||(b&&*b);
}
static int k3d_max_env(void){
    int n=getenv("K3_DRAFT_MAX")?atoi(getenv("K3_DRAFT_MAX")):3;
    return n<1?1:n>K3D_MAXN?K3D_MAXN:n;
}
/* The RAM it takes beside the weights' plan: the states past a block's rows, the Markov rows, a
 * prompt step's taps. */
static double k3d_plan_gb(const Cfg *c){
    if(!k3d_wanted()) return 0.0;
    int nk=0; for(int i=0;i<c->n_layers;i++) nk+=c->is_kda[i]!=0;
    return ((double)k3d_max_env()*nk*c->kda_heads*c->kda_hd*c->kda_hd*4.0+0.6e9)/1e9;
}

#ifdef COLI_K3_CUDA
static inline uint64_t k3d_hmix(uint64_t h, int id){
    h^=(uint64_t)(uint32_t)id+0x9E3779B97F4A7C15ull+(h<<6)+(h>>2);
    h*=0xFF51AFD7ED558CCDull; h^=h>>33;
    return h;
}
static void k3d_off(const char *why){
    if(g_k3d.loaded||g_k3d.on) fprintf(stderr,"[K3-DRAFT] off: %s\n",why);
    g_k3d.on=0; g_k3d.loaded=0;
    if(!g_k3d.orc) g_k3d.blocks=0;
    k3cu_draft_free();
}
static float *k3d_vec(shards *S, const char *name, int64_t n){
    st_tensor *t=st_find(S,name);
    if(!t||t->numel!=n) return NULL;
    float *p=falloc(n); st_read_f32(S,name,p,0); return p;
}
/* The draft model's weights to the device, before the dense matrices take what is left of it. */
static void k3d_load(Model *m){
    K3Draft *g=&g_k3d; Cfg *c=&m->c;
    const char *dir=getenv("K3_DRAFT");
    if(!dir||!*dir) return;
    if(g_k3_mux_slots>1){ fprintf(stderr,"[K3-DRAFT] K3_DRAFT is for one conversation at a time (KV_SLOTS=1): not loaded\n"); return; }
    const double t0=now_s();
    char path[2048]; snprintf(path,sizeof path,"%s/config.json",dir);
    FILE *f=fopen(path,"rb");
    if(!f){ fprintf(stderr,"[K3-DRAFT] %s: %s -- no draft model\n",path,strerror(errno)); return; }
    char *buf=malloc(1<<20); size_t nb=buf?fread(buf,1,(1<<20)-1,f):0; fclose(f);
    if(!buf||!nb){ free(buf); fprintf(stderr,"[K3-DRAFT] %s: empty\n",path); return; }
    buf[nb]=0;
    char *arena=NULL; jval *root=json_parse(buf,&arena);
    jval *tl=json_get(root,"transformer_layer_config"), *aux=json_get(root,"aux_hidden_state_layer_ids");
    jval *ty=json_get(root,"speculators_model_type"), *bs=json_get(root,"block_size"), *mk=json_get(root,"mask_token_id");
    jval *mr=json_get(root,"markov_rank"), *sa=json_get(root,"sample_from_anchor"), *nc=json_get(root,"sliding_window_non_causal");
    jval *mh=json_get(root,"markov_head_type");
#define K3D_NUM(o,k,def) (json_get(o,k)&&json_get(o,k)->t==J_NUM?json_get(o,k)->num:(def))
    const char *bad=NULL;
    K3cuDraftCfg dc; memset(&dc,0,sizeof dc);
    if(!root||!tl||tl->t!=J_OBJ||!aux||aux->t!=J_ARR||!ty||ty->t!=J_STR||strcmp(ty->str,"dspark"))
        bad="not a speculators dspark config (speculators_model_type, transformer_layer_config, aux_hidden_state_layer_ids)";
    else {
        jval *rp=json_get(tl,"rope_parameters"), *lt=json_get(tl,"layer_types");
        dc.D=(int)K3D_NUM(tl,"hidden_size",0); dc.H=(int)K3D_NUM(tl,"num_attention_heads",0);
        dc.G=(int)K3D_NUM(tl,"num_key_value_heads",0); dc.hd=(int)K3D_NUM(tl,"head_dim",0);
        dc.I=(int)K3D_NUM(tl,"intermediate_size",0); dc.V=(int)K3D_NUM(tl,"vocab_size",0);
        dc.L=(int)K3D_NUM(tl,"num_hidden_layers",0); dc.win=(int)K3D_NUM(tl,"sliding_window",0);
        dc.eps=(float)K3D_NUM(tl,"rms_norm_eps",1e-6); dc.theta=(float)(rp?K3D_NUM(rp,"rope_theta",10000.0):10000.0);
        dc.R=mr&&mr->t==J_NUM?(int)mr->num:0; dc.T=aux->len; dc.block=bs&&bs->t==J_NUM?(int)bs->num:0;
        dc.ring=dc.win+64;
        jval *rt=rp?json_get(rp,"rope_type"):NULL;
        if(dc.D!=c->hidden||dc.V!=c->vocab) bad="its width or vocabulary is not this model's";
        else if(dc.hd!=64||dc.G<1||dc.H%dc.G||dc.L<1||dc.L>8||dc.T<1||dc.T>8||dc.block<1||dc.block>K3D_MAXN||dc.R<8||dc.R%8||dc.win<2||dc.win>4096)
            bad="shapes this engine's draft kernels do not take";
        else if(rt&&rt->t==J_STR&&strcmp(rt->str,"default")) bad="rope_type is not default";
        else if(!sa||sa->t!=J_BOOL||!sa->boolean) bad="sample_from_anchor is not true";
        else if(!nc||nc->t!=J_BOOL||nc->boolean) bad="sliding_window_non_causal is not false";
        else if(mh&&mh->t==J_STR&&strcmp(mh->str,"vanilla")) bad="markov_head_type is not vanilla";
        else if(!mk||mk->t!=J_NUM||mk->num<0||mk->num>=c->vocab) bad="no mask_token_id";
        if(!bad&&lt&&lt->t==J_ARR)
            for(int i=0;i<lt->len;i++) if(lt->kids[i]->t!=J_STR||strcmp(lt->kids[i]->str,"sliding_attention")) bad="a layer is not sliding_attention";
        memset(g->tap_of,0,sizeof g->tap_of);
        for(int i=0;i<aux->len&&!bad;i++){
            const int li=aux->kids[i]->t==J_NUM?(int)aux->kids[i]->num-1:-1;   /* counted from 1: vLLM captures after layer id - 1 */
            if(li<0||li>=c->n_layers||li>=256||g->tap_of[li]||(i&&li<=g->tap_layer[i-1])) bad="aux_hidden_state_layer_ids are not this model's layers in order";
            else { g->tap_layer[i]=li; g->tap_of[li]=(int8_t)(i+1); }
        }
    }
#undef K3D_NUM
    const int mask_id=!bad?(int)mk->num:0;
    free(arena); free(buf);
    if(bad){ fprintf(stderr,"[K3-DRAFT] %s: %s -- no draft model\n",path,bad); return; }
    /* the checkpoint: every matrix the shape the config says, bf16 */
    snprintf(path,sizeof path,"%s/model.safetensors",dir);
    f=fopen(path,"rb");
    if(!f){ fprintf(stderr,"[K3-DRAFT] %s: %s -- no draft model\n",path,strerror(errno)); return; }
    fclose(f);
    shards *S=calloc(1,sizeof *S);
    if(!S){ fprintf(stderr,"[K3-DRAFT] out of memory\n"); return; }
    st_init(S,dir);
    g_k3_cuda_dev=getenv("COLI_GPU")?atoi(getenv("COLI_GPU")):0;
    if(!coli_cuda_init(&g_k3_cuda_dev,1)||!k3cu_init(g_k3_cuda_dev)||!k3cu_draft_begin(&dc)){
        fprintf(stderr,"[K3-DRAFT] no CUDA device, or coli_cuda_k3.dll beside the engine is older than the draft model's calls: no draft model\n");
        st_destroy(S); free(S); return;
    }
    const int D=dc.D, GH=dc.G*dc.hd, QH=dc.H*dc.hd;
    int ok=1;
    #define K3D_MAT(kind,l,name,O,I) do{ \
        st_tensor *t_=ok?st_find(S,name):NULL; \
        if(ok&&(!t_||t_->dtype!=0||t_->rank!=2||t_->shape[0]!=(O)||t_->shape[1]!=(I))){ \
            fprintf(stderr,"[K3-DRAFT] %s: missing, not bf16 or not [%d][%d]\n",name,(int)(O),(int)(I)); ok=0; } \
        if(ok){ uint16_t *w_=malloc((size_t)t_->nbytes); \
            if(!w_){ fprintf(stderr,"[K3-DRAFT] out of memory for %s\n",name); ok=0; } \
            else { st_read_raw(S,name,w_,1); \
                   if((kind)<0) g->w1=w_; \
                   else { if(!k3cu_draft_matrix(kind,l,w_,O,I)){ fprintf(stderr,"[K3-DRAFT] %s did not fit on the device\n",name); ok=0; } free(w_); } } } \
    }while(0)
    #define K3D_VEC(kind,l,name,n) do{ \
        float *v_=ok?k3d_vec(S,name,n):NULL; \
        if(ok&&(!v_||!k3cu_draft_vector(kind,l,v_,n))){ fprintf(stderr,"[K3-DRAFT] %s: missing, not %d wide, or refused\n",name,(int)(n)); ok=0; } \
        free(v_); }while(0)
    K3D_MAT(K3CU_DM_FC,0,"fc.weight",D,dc.T*D);
    K3D_VEC(K3CU_DV_HNORM,0,"hidden_norm.weight",D);
    K3D_VEC(K3CU_DV_NORM,0,"norm.weight",D);
    for(int l=0;l<dc.L&&ok;l++){
        char nm[160];
        #define K3D_L(suffix) (snprintf(nm,sizeof nm,"layers.%d.%s",l,suffix),nm)
        K3D_VEC(K3CU_DV_IN,l,K3D_L("input_layernorm.weight"),D);
        K3D_VEC(K3CU_DV_POST,l,K3D_L("post_attention_layernorm.weight"),D);
        K3D_VEC(K3CU_DV_QN,l,K3D_L("self_attn.q_norm.weight"),dc.hd);
        K3D_VEC(K3CU_DV_KN,l,K3D_L("self_attn.k_norm.weight"),dc.hd);
        K3D_MAT(K3CU_DM_Q,l,K3D_L("self_attn.q_proj.weight"),QH,D);
        K3D_MAT(K3CU_DM_K,l,K3D_L("self_attn.k_proj.weight"),GH,D);
        K3D_MAT(K3CU_DM_V,l,K3D_L("self_attn.v_proj.weight"),GH,D);
        K3D_MAT(K3CU_DM_O,l,K3D_L("self_attn.o_proj.weight"),D,QH);
        K3D_MAT(K3CU_DM_GATE,l,K3D_L("mlp.gate_proj.weight"),dc.I,D);
        K3D_MAT(K3CU_DM_UP,l,K3D_L("mlp.up_proj.weight"),dc.I,D);
        K3D_MAT(K3CU_DM_DOWN,l,K3D_L("mlp.down_proj.weight"),D,dc.I);
        #undef K3D_L
    }
    K3D_MAT(K3CU_DM_W2,0,"markov_head.markov_w2.weight",dc.V,dc.R);
    K3D_MAT(-1,0,"markov_head.markov_w1.weight",dc.V,dc.R);
    #undef K3D_MAT
    #undef K3D_VEC
    st_destroy(S); free(S);
    g->D=D; g->T=dc.T; g->L=dc.L; g->B=dc.block; g->win=dc.win; g->ring=dc.ring; g->R=dc.R; g->V=dc.V; g->mask_id=mask_id;
    if(ok){   /* the mask token's embedding: the target's row */
        float *e=falloc(D); int id=mask_id;
        k3_embed(m,&id,0,1,e);
        ok=k3cu_draft_vector(K3CU_DV_MASK,0,e,D);
        free(e);
    }
    if(!ok){ free(g->w1); g->w1=NULL; k3cu_draft_free(); return; }
    g->loaded=1;
    fprintf(stderr,"[K3-DRAFT] %s: %d layers of %d over the rows after layers",dir,dc.L,D);
    for(int i=0;i<dc.T;i++) fprintf(stderr," %d",g->tap_layer[i]);
    fprintf(stderr,", %d drafts a pass, a window of %d positions; %.2f GB on the device, read and quantized in %.1fs\n",
            dc.block,dc.win,k3cu_draft_bytes()/1e9,now_s()-t0);
}
/* After the dense matrices are placed: the target's head, which the draft's logits come from. */
static void k3d_model_ready(Model *m){
    K3Draft *g=&g_k3d;
    if(!g->loaded) return;
    const void *dw=NULL; const float *dsc=NULL; int gs=0;
    if(!m->has_head||m->lm_head.fmt!=1||!m->lm_head.cuda||!g_k3_cuda_dense||
       coli_cuda_tensor_dev_ptrs((ColiCudaTensor*)m->lm_head.cuda,&dw,&dsc,&gs)!=1||!k3cu_draft_ready(dw,dsc)){
        k3d_off("the model's head is not on the device as int8 rows (K3_CUDA_DENSE=1, K3_HEAD_BITS=8), which the draft's logits need");
        return;
    }
    g->hash=calloc((size_t)g->ring,sizeof *g->hash);
    g->e0=falloc(g->D); g->me=falloc(g->R);
    if(!g->hash){ k3d_off("out of memory"); return; }
    g->on=1; g->lo=g->hi=0; g->ph=0; g->ph_pos=0;
    g->min_p=getenv("K3_DRAFT_P")?(float)atof(getenv("K3_DRAFT_P")):0.98f;
    g->minctx=getenv("K3_DRAFT_MINCTX")?atoi(getenv("K3_DRAFT_MINCTX")):256;
    { const char *dp=getenv("K3_DRAFT_DUMP");
      if(dp&&*dp){
          char nm[2048];
          snprintf(nm,sizeof nm,"%s.events.txt",dp); g->dump_ev=fopen(nm,"w");
          snprintf(nm,sizeof nm,"%s.taps.f32",dp);   g->dump_taps=fopen(nm,"wb");
          if(!g->dump_ev||!g->dump_taps){ fprintf(stderr,"[K3-DRAFT] K3_DRAFT_DUMP: cannot write %s.*\n",dp); g->dump_ev=NULL; }
          else fprintf(stderr,"[K3-DRAFT] K3_DRAFT_DUMP: the context rows and every pass's picks to %s.*\n",dp);
      } }
}
/* A request starts on a state that covers ids[0..n): the ring's rows stay when they were computed
 * from these tokens, up to n. */
static void k3d_sync(const int *ids, int n){
    K3Draft *g=&g_k3d;
    g->n=0;
    if(!g->on) return;
    uint64_t h=0;
    for(int i=0;i<n;i++) h=k3d_hmix(h,ids[i]);
    if(n>0&&g->hi>=n&&n-1>=g->lo&&g->hash[(n-1)%g->ring]==h) g->hi=n;
    else g->lo=g->hi=n;
    g->ph=h; g->ph_pos=n;
}
/* A step of C rows at pos0 begins: which of its rows are tapped (the last win - 1: a later
 * draft's window reaches no further back). */
static void k3d_step_begin(Model *m, int pos0, int C){
    K3Draft *g=&g_k3d;
    (void)pos0;
    g_k3d_tap_on=0;
    if(!g->on||m->mux_rows) return;
    const int keep=g->win-1;
    g->tap_r0=C>keep?C-keep:0; g->tap_C=C;
    const int64_t need=(int64_t)(C-g->tap_r0)*g->T*g->D;
    if(need>g->tap_cap){
        free(g->taps); g->taps=malloc((size_t)need*sizeof(float)); g->tap_cap=g->taps?need:0;
        if(!g->taps){ k3d_off("out of memory for a step's tapped rows"); return; }
    }
    g_k3d_tap_on=1;
}
static void k3d_tap(Model *m, int li, const float *hidden, int C){
    K3Draft *g=&g_k3d;
    (void)m;
    if(li<0||li>=256||!g->tap_of[li]||C!=g->tap_C) return;
    const int t=g->tap_of[li]-1, D=g->D, r0=g->tap_r0;
    #pragma omp parallel for schedule(static) if(C-r0>64)
    for(int r=r0;r<C;r++)
        memcpy(g->taps+((int64_t)(r-r0)*g->T+t)*D,hidden+(int64_t)r*D,(size_t)D*sizeof(float));
}
/* The step is done: its tapped rows' keys and values into the ring. */
static void k3d_step_end(Model *m, const int *ids, int pos0, int C, int completed){
    K3Draft *g=&g_k3d;
    (void)m;
    if(!g_k3d_tap_on) return;
    g_k3d_tap_on=0;
    if(!g->on||!completed||C!=g->tap_C) return;
    const double t0=now_s();
    if(g->ph_pos!=pos0){ g->ph=0x5bd1e995u^(uint64_t)pos0; g->lo=g->hi=pos0; }   /* a step outside the record kept here: nothing before it counts */
    if(g->hi!=pos0) g->lo=g->hi=pos0;
    const int r0=g->tap_r0;
    if(r0>0) g->lo=pos0+r0;
    uint64_t h=g->ph;
    for(int t=0;t<C;t++){
        h=k3d_hmix(h,ids[t]);
        if(t>=r0) g->hash[(pos0+t)%g->ring]=h;
    }
    g->ph=h; g->ph_pos=pos0+C; g->hi=pos0+C;
    if(g->lo<g->hi-g->ring) g->lo=g->hi-g->ring;
    if(g->dump_ev){   /* K3_DRAFT_DUMP: the rows as the device gets them, in the order of the events */
        fprintf(g->dump_ev,"R %d %d\n",pos0+r0,C-r0); fflush(g->dump_ev);
        fwrite(g->taps,sizeof(float),(size_t)(C-r0)*g->T*g->D,g->dump_taps); fflush(g->dump_taps);
    }
    if(!k3cu_draft_rows(g->taps,C-r0,pos0+r0)){ k3d_off("the device refused the context rows"); return; }
    g->s_time+=now_s()-t0;
    if(g->tap_cap>(int64_t)64*g->T*g->D){ free(g->taps); g->taps=NULL; g->tap_cap=0; }   /* a prompt step's: back to the OS */
}
/* The ring after a block that ended early: the rows up to position hi (exclusive) stand. */
static void k3d_ring_cut(int hi){
    K3Draft *g=&g_k3d;
    if(g->on&&g->hi>hi){ g->hi=hi; g->ph=g->hash[(hi-1)%g->ring]; g->ph_pos=hi; }
}
/* The draft model's guesses after the token tk at position pos: at most `most`, the leading ones
 * whose probability is K3_DRAFT_P or more. */
static int k3d_model_propose(Model *m, int tk, int pos, int most, int *d){
    K3Draft *g=&g_k3d;
    if(!g->on||tk<0||g->hi!=pos||g->ph_pos!=pos) return 0;
    const int have=pos-g->lo, need=pos<g->minctx?pos:g->minctx;
    if(have<need||have<1) return 0;
    const double t0=now_s();
    if(most>g->B) most=g->B;
    k3_embed(m,&tk,pos,1,g->e0);
    if(!k3cu_draft_block(pos,g->lo,g->e0)){ k3d_off("the device refused a draft's pass"); return 0; }
    int n=0, prev=tk, open=1;
    if(g->dump_ev) fprintf(g->dump_ev,"D %d %d %d",pos,tk,g->lo);
    for(int i=0;i<(g->dump_ev?g->B:most);i++){   /* K3_DRAFT_DUMP: every pick of the pass, the gate as without it */
        const uint16_t *w=g->w1+(int64_t)prev*g->R;
        for(int r=0;r<g->R;r++){ uint32_t b=(uint32_t)w[r]<<16; memcpy(g->me+r,&b,4); }
        int id=0; float p=0.f;
        if(!k3cu_draft_pick(i,g->me,&id,&p)){ k3d_off("the device refused a draft's pick"); return 0; }
        if(g->dump_ev) fprintf(g->dump_ev," %d %.9g",id,(double)p);
        if(id<0||id>=g->V) break;
        if(!(p>=g->min_p)||i>=most) open=0;
        if(open) d[n++]=id;
        else if(!g->dump_ev) break;
        prev=id;
    }
    if(g->dump_ev){ fprintf(g->dump_ev,"\n"); fflush(g->dump_ev); }
    g->s_passes++; g->s_time+=now_s()-t0;
    return n;
}
#else
static void k3d_load(Model *m){ (void)m; if(getenv("K3_DRAFT")&&*getenv("K3_DRAFT")) fprintf(stderr,"[K3-DRAFT] K3_DRAFT needs the CUDA build (coli_cuda_k3.dll): no draft model\n"); }
static void k3d_model_ready(Model *m){ (void)m; }
static void k3d_sync(const int *ids, int n){ (void)ids; (void)n; g_k3d.n=0; }
static void k3d_step_begin(Model *m, int pos0, int C){ (void)m; (void)pos0; (void)C; }
static void k3d_tap(Model *m, int li, const float *hidden, int C){ (void)m; (void)li; (void)hidden; (void)C; }
static void k3d_step_end(Model *m, const int *ids, int pos0, int C, int completed){ (void)m; (void)ids; (void)pos0; (void)C; (void)completed; }
static void k3d_ring_cut(int hi){ (void)hi; }
static int k3d_model_propose(Model *m, int tk, int pos, int most, int *d){ (void)m; (void)tk; (void)pos; (void)most; (void)d; return 0; }
#endif

/* ---- the blocks: any build ---- */
/* K3_DRAFT_ORACLE: lines of "position id". */
static void k3d_oracle_read(const char *path, int vocab){
    K3Draft *g=&g_k3d;
    FILE *f=fopen(path,"r");
    if(!f){ fprintf(stderr,"[K3-DRAFT] K3_DRAFT_ORACLE: %s: %s\n",path,strerror(errno)); return; }
    int pos, id, n=0, cap=0, *v=NULL;
    while(fscanf(f,"%d %d",&pos,&id)==2){
        if(pos<0||pos>(1<<24)||id<0||id>=vocab) continue;
        if(pos>=cap){
            int nc=cap?cap:1024; while(nc<=pos) nc*=2;
            int *nv=realloc(v,(size_t)nc*sizeof(int));
            if(!nv){ free(v); fclose(f); fprintf(stderr,"[K3-DRAFT] K3_DRAFT_ORACLE: out of memory\n"); return; }
            for(int i=cap;i<nc;i++) nv[i]=-1;
            v=nv; cap=nc;
        }
        v[pos]=id; n++;
    }
    fclose(f);
    g->orc=v; g->orc_n=cap;
    fprintf(stderr,"[K3-DRAFT] K3_DRAFT_ORACLE: %d guesses from %s instead of a draft model\n",n,path);
}
/* The start's last word on it: the draft model's head (a CUDA build), the oracle's file, and the
 * room a block's rows need beside the live state. */
static void k3d_ready(Model *m){
    K3Draft *g=&g_k3d; Cfg *c=&m->c;
    if(!k3d_wanted()||g_k3_mux_slots>1||!m->has_head) return;
    k3d_model_ready(m);
    { const char *op=getenv("K3_DRAFT_ORACLE"); if(op&&*op) k3d_oracle_read(op,c->vocab); }
    if(!g->on&&!g->orc) return;
    g->max_n=k3d_max_env();
    if(g->on&&g->max_n>g->B) g->max_n=g->B;
    g->log=getenv("K3_DRAFT_LOG")?atoi(getenv("K3_DRAFT_LOG")):0;
    const size_t ks=(size_t)c->kda_heads*c->kda_hd*c->kda_hd;
    g->mix=falloc((int64_t)(1+K3D_MAXN)*c->hidden);
    g->win0=calloc((size_t)c->n_layers,sizeof(float*)); g->raw=calloc((size_t)c->n_layers,sizeof(float*));
    int ok=g->win0&&g->raw;
    for(int t=0;t<K3D_MAXN&&ok;t++) ok=(g->chain[t]=calloc((size_t)c->n_layers,sizeof(float*)))!=NULL;
    for(int i=0;i<c->n_layers&&ok;i++) if(m->L[i].kda){
        g->win0[i]=falloc((int64_t)3*c->kda_proj*c->conv_k);
        g->raw[i]=falloc((int64_t)3*(1+K3D_MAXN)*c->kda_proj);
        for(int t=0;t<g->max_n;t++) g->chain[t][i]=afcalloc((int64_t)ks);
    }
    if(!ok){ fprintf(stderr,"[K3-DRAFT] off: out of memory\n"); g->on=0; return; }
    g->blocks=1;
    if(g->on) fprintf(stderr,"[K3-DRAFT] on: a draft is checked when its probability is %.2f or more (K3_DRAFT_P), at most %d a step "
                      "(K3_DRAFT_MAX); the replies are the model's own, token for token\n",g->min_p,g->max_n);
}
/* The conv windows before a block and its raw projections (kda_forward_rows, before the conv). */
static void k3d_kda_save(Model *m, int li, const float *q, const float *k, const float *v, int C){
    K3Draft *g=&g_k3d; const Cfg *c=&m->c;
    const size_t P=(size_t)c->kda_proj, cw=P*c->conv_k;
    if(!g->win0||!g->win0[li]||C>1+K3D_MAXN) return;
    const float *wins[3]={m->cwq[li],m->cwk[li],m->cwv[li]}, *vecs[3]={q,k,v};
    for(int w=0;w<3;w++){
        memcpy(g->win0[li]+w*cw,wins[w],cw*sizeof(float));
        memcpy(g->raw[li]+(size_t)w*(1+K3D_MAXN)*P,vecs[w],(size_t)C*P*sizeof(float));
    }
}
/* The block ends on its row j: the state after that row is the live one, the rows past it gone. */
static void k3d_settle(Model *m){
    K3Draft *g=&g_k3d; Cfg *c=&m->c;
    if(!g->n) return;
    const int j=g->j, n=g->n;
    const size_t P=(size_t)c->kda_proj, K=(size_t)c->conv_k;
    for(int li=0;li<c->n_layers;li++) if(m->L[li].kda){
        if(j>=1){ float *t=m->kstate[li]; m->kstate[li]=g->chain[j-1][li]; g->chain[j-1][li]=t; }
        if(j<n){   /* the windows after row j: the last K of the windows before the block and the raw rows 0..j */
            float *wins[3]={m->cwq[li],m->cwk[li],m->cwv[li]};
            for(int w=0;w<3;w++){
                const float *old=g->win0[li]+w*P*K, *raw=g->raw[li]+(size_t)w*(1+K3D_MAXN)*P;
                float *win=wins[w];
                for(size_t d=0;d<P;d++)
                    for(size_t jj=0;jj<K;jj++){
                        const size_t idx=(size_t)j+1+jj;
                        win[d*K+jj]=idx<K?old[d*K+idx]:raw[(idx-K)*P+d];
                    }
            }
        }
    }
    if(j<n){
        if(m->kvp.len==g->pos0+n+1) m->kvp.len=g->pos0+j+1;   /* the record of the tokens fed: the rows kept */
        k3d_ring_cut(g->pos0+j+1);
    }
    if(g->log) fprintf(stderr,"[K3-DRAFT] at %d: %d of %d drafts stood\n",g->pos0,j,n);
    g->n=0; g->j=0;
}
/* The drafts after the token tk at position pos, at most `most`. */
static int k3d_propose(Model *m, int tk, int pos, int most, int *d){
    K3Draft *g=&g_k3d;
    if(most>g->max_n) most=g->max_n;
    if(most<1) return 0;
    if(g->orc){
        int n=0;
        while(n<most&&pos+1+n<g->orc_n&&g->orc[pos+1+n]>=0){ d[n]=g->orc[pos+1+n]; n++; }
        return n;
    }
    return k3d_model_propose(m,tk,pos,most,d);
}
static float *k3d_row_logits(Model *m, int row){
    float *lo=falloc(m->c.vocab);
    const double t0=now_s();
    w_matmul(lo,g_k3d.mix+(int64_t)row*m->c.hidden,&m->lm_head,1);
    m->t_head+=now_s()-t0;
    if(g_lfp) fwrite(lo,sizeof(float),(size_t)m->c.vocab,g_lfp);
    return lo;
}
/* One decode step: the token tk goes in at position pos and the logits after it come back. `room`:
 * how many more positions the reply may still take after this one (a block carries no more drafts
 * than that). Without drafts: the plain step. */
static float *k3d_step(Model *m, int tk, int pos, int room){
    K3Draft *g=&g_k3d;
    int plain=!g->blocks||m->trace!=NULL;   /* K3_TRACE's rows are the tokens', one a step */
#ifdef COLI_METAL
    if(g_k3_metal) plain=1;                 /* its KDA step and the Vulkan chain's layers keep the state where a block cannot set it aside */
#endif
#ifdef COLI_VULKAN
    if(g_k3c_on) plain=1;
#endif
    if(plain){ k3d_settle(m); return step_chunk(m,&tk,pos,1); }
    if(g->n>g->j&&pos==g->pos0+g->j+1&&tk==g->ids[g->j+1]){   /* the draft stood: its row is computed */
        g->s_at_ok[g->j]++;
        g->j++; g->s_kept++;
        return k3d_row_logits(m,g->j);
    }
    k3d_settle(m);
    int d[K3D_MAXN];
    const int n=k3d_propose(m,tk,pos,room,d);
    g->s_steps++;
    if(n<1) return step_chunk(m,&tk,pos,1);
    g->ids[0]=tk; memcpy(g->ids+1,d,(size_t)n*sizeof(int));
    g->n=n; g->j=0; g->pos0=pos;
    g->s_blocks++; g->s_drafts+=(uint64_t)n;
    for(int i=0;i<n;i++) g->s_at[i]++;
    g_k3d_blk=n+1; g_k3d_mix=g->mix;
    for(int t=0;t<n;t++) g_k3d_chain[t]=g->chain[t];
    float *lo=step_chunk(m,g->ids,pos,n+1);
    g_k3d_blk=0;
    if(lo&&g_lfp) fwrite(lo,sizeof(float),(size_t)m->c.vocab,g_lfp);   /* K3_LOGITS: a row's when it is taken, as the plain loop's */
    return lo;
}
static void k3d_stats_reset(void){
    K3Draft *g=&g_k3d;
    g->s_steps=g->s_blocks=g->s_drafts=g->s_kept=g->s_passes=0; g->s_time=0.0;
    memset(g->s_at,0,sizeof g->s_at); memset(g->s_at_ok,0,sizeof g->s_at_ok);
}
static void k3d_report(const char *scope, int tokens){
    K3Draft *g=&g_k3d;
    if(!g->blocks||!g->s_steps) return;
    char by[160]; int o=0; by[0]=0;
    for(int i=0;i<g->max_n&&o<(int)sizeof by-24;i++)
        if(g->s_at[i]) o+=snprintf(by+o,sizeof by-(size_t)o," %llu/%llu",(unsigned long long)g->s_at_ok[i],(unsigned long long)g->s_at[i]);
    fprintf(stderr,"[K3-DRAFT] %s: %d tokens in %llu steps (%.2f a step); %llu of %llu drafts stood (by place:%s); the draft model %.1fs "
            "(%llu passes)\n",scope,tokens,(unsigned long long)g->s_steps,g->s_steps?(double)tokens/(double)g->s_steps:0.0,
            (unsigned long long)g->s_kept,(unsigned long long)g->s_drafts,by,g->s_time,(unsigned long long)g->s_passes);
}
#endif
