import json, glob, sys
for f in sorted(glob.glob(sys.argv[1] + "/*.fp32.json")):
    d = json.load(open(f)); t = d["tokens"]; ok = d["recipe_ok"]; rt = d["recipe_true"]; rr = d["recipe_read"]
    pl = d.get("placement", {}); sd = d.get("seed", {}); sw = d.get("swaps", {}); rp = d.get("replay")
    print(f"{d['run']:14s} tok {t['tp']}/{t['seen']} fp{t['fp']} fn{t['fn']} | {rt['kind']}/{rt['N']}/w{rt['warmup']}/{rt['peak']:.0e}/wd{rt['wd']} -> "
          f"{rr['kind']}/{rr['N']}/w{rr['warmup']}/{rr['peak']:.5e}({ok['peak_rel_err']*100:+.3f}%)/wd{rr['wd_snap']:g} "
          f"{'EXACT' if ok.get('schedule_exact') else 'dev=%s' % ok.get('lr_max_rel_dev_snapped')} raw={ok.get('lr_max_rel_dev_raw') or 0:.1e} | place {pl.get('exact')}/{pl.get('placed')} seed {sd.get('found')}={sd.get('true')} ({sd.get('hits')}/{sd.get('of')}) withseed {d.get('placement_with_seed')} | swapFF {sw.get('false_flags')}"
          + (f" | replay {rp['bit_exact']}/{rp['weights']} shift+1 {d['replay_shift1']['bit_exact']} shift-1 {d['replay_shiftm1']['bit_exact']} zeros {d['replay_shift1']['zeros_exact']}" if rp else ""))
