/* k3_mirror.h -- Kimi K3's weights whole on two drives (K3_MIRROR, K3_FILL), included by kimi_k3.c.
 *
 * tools/k3_download.py cuts every MoE layer's shard over the drives in proportion to their
 * speeds (on the 4090 machine 61% of each layer's experts on C:, the rest on D:), so an expert
 * is read from one drive, and a layer waits for whichever drive got more of its reads. With
 * every file on both drives -- the model's folder and a K3_DIRS folder each holding all of them
 * -- an expert is read from both at once instead: the first K3_MIRROR_SPLIT of its bytes from the
 * model's folder, the rest from the copy on the other drive (k3_io_submit cuts the read in two,
 * each drive's readers take their piece). A layer's last read lands sooner, and neither drive
 * idles while the other still has reads queued.
 *
 * Nothing read changes: a copy counts only when its size and its safetensors header are the
 * model folder's file's (st_mirror_add's rule: the same header names the same offsets) and it is
 * on another device (the same drive twice adds no speed, and the cut layout's own parts, found
 * through K3_DIRS, are not their own copies). Every file that has no such copy is read from
 * where it is, as before. K3_MIRROR=0: no copy is used.
 *
 * K3_FILL=1: at the start, before the shards are indexed, every *.safetensors of the K3_DIRS
 * folders that the model's folder lacks is copied there (as <name>.k3fill, renamed when it is
 * whole), so the model's folder holds every file again after some were deleted, while the drive
 * keeps K3_FILL_KEEP_GB free (64); what does not fit is read from the other drive. A file of the
 * same name that differs is left alone and named; one with the same header but shorter is an
 * interrupted copy and is copied again. */
#ifndef K3_MIRROR_H
#define K3_MIRROR_H

#ifndef _WIN32
#include <sys/statvfs.h>
#endif

static int g_k3_mirror_on;                  /* copies registered: an expert's read is cut in two */
static double g_k3_msplit = 0.6;            /* K3_MIRROR_SPLIT: the model folder's share of each read */
static char *g_k3_mpath[ST_MAX_SHARDS];     /* shard fi's copy on another drive; NULL: none */
/* A large step (more than K3_MIRROR_ROWS rows, 16: a prompt's) reads each expert whole, from the drive
 * whose queue is shorter for its speed: such a step keeps both drives busy whatever it does, and two
 * half-size reads take 3-4% longer than one (measured, a 3,759-token prompt). A small step's reads --
 * a decode step's and its drafts' -- are cut in two: its layer waits for the last of a few of them.
 * K3_MIRROR_ROWS=-1: every read cut; 0: every read whole. moe_forward sets it for its step. */
static int g_k3_mirror_rows = 16;
static volatile int g_k3_io_whole;
/* Every file a routed expert is read from has a copy (k3_io_start): only then is a small step's read cut
 * in two. With copies of some files only -- the cut layout after K3_FILL brought D:'s parts to C:, D:
 * still lacking C:'s -- the files without one keep all their reads on their drive, and cutting the others
 * 60/40 would pile 85% of a decode token's bytes on C: (61% in the cut layout); there every read of a file
 * with a copy goes whole to the drive whose queue is shorter, which balances around them instead. */
static int g_k3_mirror_full;
/* K3_MIRROR_ANY=1 (tests, a machine with one drive): a folder on the model's own drive counts as
 * another drive -- its copies are used and filled from, and get readers and queues of their own */
static int k3m_any(void){ const char *e=getenv("K3_MIRROR_ANY"); return e&&atoi(e)>0; }

/* the folders of a ';'- or ','-separated list, trimmed, into out[] (at most n): how many */
static int k3m_dirs(const char *list, char out[][1024], int n){
    int k=0;
    for(const char *p=list?list:""; *p&&k<n; ){
        const char *e=p;
        while(*e&&*e!=';'&&*e!=',') e++;
        const char *a=p, *b=e;
        while(a<b&&*a==' ') a++;
        while(b>a&&b[-1]==' ') b--;
        if(b>a){
            size_t len=(size_t)(b-a); if(len>1023) len=1023;
            memcpy(out[k],a,len); out[k][len]=0; k++;
        }
        p=*e?e+1:e;
    }
    return k;
}

static const char *k3m_base(const char *path){
    const char *b=strrchr(path,'/');
#ifdef _WIN32
    const char *b2=strrchr(path,'\\'); if(b2&&(!b||b2>b)) b=b2;
#endif
    return b?b+1:path;
}

/* a safetensors file's size and header (its 8-byte length and the JSON after it); 0 when the
 * file is missing or has no readable header */
typedef struct { int64_t size; uint64_t hlen; char *hdr; } K3mHead;
static int k3m_head(const char *path, K3mHead *h){
    memset(h,0,sizeof *h);
    int fd=open(path,COMPAT_O_RDONLY);
    if(fd<0) return 0;
    struct stat sb;
    int ok=fstat(fd,&sb)==0;
    if(ok){
        h->size=(int64_t)sb.st_size;
        ok=h->size>=8&&pread(fd,&h->hlen,8,0)==8&&h->hlen>0&&h->hlen<=(uint64_t)ST_MAX_HEADER&&
           (int64_t)(8+h->hlen)<=h->size&&(h->hdr=malloc(h->hlen))!=NULL&&
           pread(fd,h->hdr,(size_t)h->hlen,8)==(ssize_t)h->hlen;
    }
    close(fd);
    if(!ok){ free(h->hdr); h->hdr=NULL; }
    return ok;
}
static int k3m_same_head(const K3mHead *a, const K3mHead *b){
    return a->hlen==b->hlen&&!memcmp(a->hdr,b->hdr,(size_t)a->hlen);
}

/* bytes free for this process on the drive of `dir`; -1 when unknown */
static double k3m_free_bytes(const char *dir){
#ifdef _WIN32
    ULARGE_INTEGER avail;
    return GetDiskFreeSpaceExA(dir,&avail,NULL,NULL)?(double)avail.QuadPart:-1.0;
#else
    struct statvfs v;
    return statvfs(dir,&v)==0?(double)v.f_bavail*(double)v.f_frsize:-1.0;
#endif
}

#ifdef _WIN32
typedef struct { const char *name; double t_shown, t0; } K3mProg;
static DWORD CALLBACK k3m_progress(LARGE_INTEGER total, LARGE_INTEGER done, LARGE_INTEGER sz, LARGE_INTEGER sd,
                                   DWORD sn, DWORD why, HANDLE hs, HANDLE hd, LPVOID arg){
    (void)sz; (void)sd; (void)sn; (void)why; (void)hs; (void)hd;
    K3mProg *p=arg;
    double t=now_s();
    if(t-p->t_shown>=10.0){
        p->t_shown=t;
        fprintf(stderr,"[K3-FILL]   %s: %.1f of %.1f GB (%.1f GB/s)\n",p->name,done.QuadPart/1e9,total.QuadPart/1e9,
                done.QuadPart/1e9/(t-p->t0+1e-9));
        fflush(stderr);
    }
    return PROGRESS_CONTINUE;
}
#endif

/* src -> dst through dst.k3fill: 1 when dst is the whole copy */
static int k3m_copy(const char *src, const char *dst, const char *name){
    char tmp[2100]; snprintf(tmp,sizeof tmp,"%s.k3fill",dst);
#ifdef _WIN32
    K3mProg p={name,now_s(),now_s()};
    if(!CopyFileExA(src,tmp,k3m_progress,&p,NULL,COPY_FILE_NO_BUFFERING)){
        fprintf(stderr,"[K3-FILL] copying %s failed (Windows error %lu)\n",src,(unsigned long)GetLastError());
        DeleteFileA(tmp); return 0;
    }
    if(!MoveFileExA(tmp,dst,MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)){
        fprintf(stderr,"[K3-FILL] naming %s failed (Windows error %lu)\n",dst,(unsigned long)GetLastError());
        DeleteFileA(tmp); return 0;
    }
    return 1;
#else
    (void)name;
    int in=open(src,O_RDONLY), out=in<0?-1:open(tmp,O_WRONLY|O_CREAT|O_TRUNC,0644);
    size_t cap=(size_t)64<<20; char *buf=out<0?NULL:malloc(cap);
    int ok=buf!=NULL;
    while(ok){
        ssize_t r=read(in,buf,cap);
        if(r<0&&errno==EINTR) continue;
        if(r<=0){ ok=r==0; break; }
        for(ssize_t w=0;ok&&w<r;){
            ssize_t k=write(out,buf+w,(size_t)(r-w));
            if(k<0&&errno==EINTR) continue;
            if(k<=0) ok=0; else w+=k;
        }
    }
    free(buf);
    if(out>=0&&fsync(out)!=0) ok=0;
    if(out>=0) close(out);
    if(in>=0) close(in);
    if(ok&&rename(tmp,dst)!=0) ok=0;
    if(!ok){ fprintf(stderr,"[K3-FILL] copying %s failed: %s\n",src,strerror(errno)); unlink(tmp); }
    return ok;
#endif
}

/* K3_FILL=1: the model's folder gets every *.safetensors of the K3_DIRS folders it lacks */
static void k3_fill(const char *snap){
    const char *on=getenv("K3_FILL");
    if(!on||!atoi(on)) return;
    char (*dirs)[1024]=malloc(8*sizeof *dirs);
    if(!dirs){ fprintf(stderr,"OOM fill\n"); exit(1); }
    const int nd=k3m_dirs(getenv("K3_DIRS"),dirs,8);
    const double keep=(getenv("K3_FILL_KEEP_GB")?atof(getenv("K3_FILL_KEEP_GB")):64.0)*1e9;
    const double t0=now_s();
    struct stat ss;
    if(stat(snap,&ss)!=0){ free(dirs); return; }
    {   /* what an interrupted copy left (only this file's own suffix) */
        DIR *d=opendir(snap); struct dirent *e;
        while(d&&(e=readdir(d))){
            size_t n=strlen(e->d_name);
            if(n>7&&!strcmp(e->d_name+n-7,".k3fill")){
                char p[2100]; snprintf(p,sizeof p,"%s/%s",snap,e->d_name);
                if(!unlink(p)) fprintf(stderr,"[K3-FILL] removed %s, an interrupted copy\n",p);
            }
        }
        if(d) closedir(d);
    }
    int checked=0, copied=0, left=0, differ=0;
    double bytes=0, left_bytes=0;
    for(int k=0;k<nd;k++){
        struct stat ds;
        if(stat(dirs[k],&ds)!=0){ fprintf(stderr,"[K3-FILL] %s: not there\n",dirs[k]); continue; }
        if(ds.st_dev==ss.st_dev&&!k3m_any()) continue;   /* the model's own drive: nothing to gain */
        DIR *d=opendir(dirs[k]); struct dirent *e;
        while(d&&(e=readdir(d))){
            const char *dot=strrchr(e->d_name,'.');
            if(!dot||strcmp(dot,".safetensors")) continue;
            char src[2100], dst[2100];
            snprintf(src,sizeof src,"%s/%s",dirs[k],e->d_name);
            snprintf(dst,sizeof dst,"%s/%s",snap,e->d_name);
            K3mHead hs, hd;
            if(!k3m_head(src,&hs)){ fprintf(stderr,"[K3-FILL] %s: no safetensors header, not copied\n",src); continue; }
            checked++;
            const int have=k3m_head(dst,&hd);
            const int same=have&&hd.size==hs.size&&k3m_same_head(&hd,&hs);
            const int cut_short=have&&!same&&hd.size<hs.size&&k3m_same_head(&hd,&hs);
            free(hd.hdr);
            if(same){ free(hs.hdr); continue; }
            if(have&&!cut_short){
                fprintf(stderr,"[K3-FILL] %s differs from %s (its size or header): left as it is, read from where each is\n",dst,src);
                differ++; free(hs.hdr); continue;
            }
            free(hs.hdr);
            const double fb=k3m_free_bytes(snap);
            if(fb>=0&&fb-(double)hs.size<keep){
                left++; left_bytes+=(double)hs.size;
                continue;
            }
            fprintf(stderr,"[K3-FILL] %s %s (%.1f GB) from %s\n",cut_short?"copying again":"copying",e->d_name,hs.size/1e9,dirs[k]);
            fflush(stderr);   /* a redirected stderr is buffered (msvcrt): the copies take minutes */
            if(k3m_copy(src,dst,e->d_name)){ copied++; bytes+=(double)hs.size; }
        }
        if(d) closedir(d);
    }
    const double dt=now_s()-t0;
    if(copied) fprintf(stderr,"[K3-FILL] copied %d file(s), %.1f GB in %.0f s (%.1f GB/s)\n",copied,bytes/1e9,dt,bytes/1e9/(dt+1e-9));
    if(left) fprintf(stderr,"[K3-FILL] %d file(s), %.1f GB, not copied: %s would keep less than %.0f GB free; "
                            "they are read from the other drive\n",left,left_bytes/1e9,snap,keep/1e9);
    fprintf(stderr,"[K3-FILL] %s: %d of the other drive's %d model file(s) here%s (checked in %.1f s)\n",snap,
            checked-left-differ,checked,differ?", some differ":"",dt);
    fflush(stderr);
    free(dirs);
}

/* After st_init_multi: every shard file of the model's folder whose copy is on another drive (a
 * K3_DIRS folder, the same size and header) gets that copy as replica 1 (st.h's mirror fds, for
 * the readers' pread elsewhere than Windows; the readers on Windows open their own handles). */
static void k3_mirror_setup(Model *m){
    const char *on=getenv("K3_MIRROR");
    if(on&&!atoi(on)) return;
    char (*dirs)[1024]=malloc(8*sizeof *dirs);
    if(!dirs){ fprintf(stderr,"OOM mirror\n"); exit(1); }
    const int nd=k3m_dirs(getenv("K3_DIRS"),dirs,8);
    shards *S=&m->S;
    if(!nd||S->nrep){ free(dirs); return; }
    for(int i=0;i<ST_MAX_SHARDS;i++){ S->mfds[0][i]=-1; S->mdfds[0][i]=-1; }
    int n=0, other=0;
    double bytes=0;
    for(int i=0;i<S->nfd&&i<ST_MAX_SHARDS;i++){
        struct stat ps;
        if(stat(S->paths[i],&ps)!=0) continue;
        K3mHead hp; int have_hp=0;
        for(int k=0;k<nd;k++){
            char mp[2100]; snprintf(mp,sizeof mp,"%s/%s",dirs[k],k3m_base(S->paths[i]));
            struct stat ms;
            if(stat(mp,&ms)!=0) continue;
            if(!strcmp(mp,S->paths[i])) continue;                 /* the very file (the cut layout's own part) */
            if(ms.st_dev==ps.st_dev&&!k3m_any()) continue;        /* the same drive: no speed to gain */
            K3mHead hm;
            if(!have_hp&&!(have_hp=k3m_head(S->paths[i],&hp))) break;
            if(!k3m_head(mp,&hm)) continue;
            const int same=hm.size==S->sizes[i]&&hp.size==S->sizes[i]&&k3m_same_head(&hm,&hp);
            free(hm.hdr);
            if(!same){ other++; continue; }
#ifndef _WIN32
            /* the readers read through these; on Windows each opens its own (Windows) handles of
             * the copy instead, and msvcrt's 2,048 C descriptors are kept for the rest */
            int fd=open(mp,COMPAT_O_RDONLY);
            if(fd<0) continue;
            S->mfds[0][i]=fd;
            S->mdfds[0][i]=compat_open_direct(mp);
#endif
            g_k3_mpath[i]=strdup(mp);
            n++; bytes+=(double)S->sizes[i];
            break;
        }
        if(have_hp) free(hp.hdr);
    }
    free(dirs);
    if(other) fprintf(stderr,"[K3-MIRROR] %d file(s) of the same name on another drive differ in size or header: not used\n",other);
    if(!n) return;
#ifndef _WIN32
    S->nmirror[0]=n; S->nrep=1;
#endif
    g_k3_mirror_on=1;
    const char *sp=getenv("K3_MIRROR_SPLIT");
    if(sp&&atof(sp)>0.02&&atof(sp)<0.98) g_k3_msplit=atof(sp);
    const char *rows=getenv("K3_MIRROR_ROWS");
    if(rows&&*rows) g_k3_mirror_rows=atoi(rows);
    char whole[160]="";
    if(g_k3_mirror_rows>=0)
        snprintf(whole,sizeof whole,"; in a step of more than %d rows each whole, from the drive whose queue is shorter (K3_MIRROR_ROWS)",
                 g_k3_mirror_rows);
    fprintf(stderr,"[K3-MIRROR] %d of %d shard files (%.0f GB) also on another drive: an expert's read takes %.0f%% "
                   "of its bytes from the model's folder and %.0f%% from the copy at once (K3_MIRROR_SPLIT)%s\n",
            n,S->nfd,bytes/1e9,100.0*g_k3_msplit,100.0*(1.0-g_k3_msplit),whole);
}

/* where a read of dlen bytes (a 4 KiB multiple) is cut: a 4 KiB multiple inside it, or dlen for a
 * read of less than two blocks (the first piece is all of it, the second only the tail past it) */
static int64_t k3_mirror_cut(int64_t dlen){
    if(dlen<8192) return dlen>0?dlen:0;
    int64_t c=(int64_t)((double)dlen*g_k3_msplit)&~(int64_t)4095;
    if(c<4096) c=4096;
    if(c>dlen-4096) c=dlen-4096;
    return c;
}
#endif
