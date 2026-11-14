// Evaluate this function in functions.exec, supplying the committed manifest.
// GitHub calls use the connected plugin. No credentials or network bypass.
async function publishCrossBehavior(m) {
  const repo = "LemonQC/tempcross";
  const branch = "cross-behavior-task-expansion";
  if (m.branch !== branch) throw Error("Refuse publication outside the authorized branch");
  const quote = s => "'" + s.replace(/'/g, "'\\''") + "'";
  const directory = "/home/users/lemon/cross-behavior-20261114/publications/" + m.local_commit;
  const statePath = directory + "/api_state.json";
  const shell = async (cmd, budget=1500) => {
    const r = await tools.exec_command({cmd, workdir:"/home/users/lemon/CodexProject/tempcross-cross-behavior", max_output_tokens:budget});
    if (r.exit_code !== 0) throw Error(r.output);
    return r.output;
  };
  const save = async state => {
    await shell("python -c 'import pathlib,sys;pathlib.Path(sys.argv[1]).write_text(sys.argv[2])' " + quote(statePath) + " " + quote(JSON.stringify(state)));
  };
  const initial = JSON.parse(await shell("python -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); print(p.read_text() if p.exists() else \"{}\")' " + quote(statePath),100000));
  const state = {blobs:[], ...initial};
  const head = async () => {
    const r = await tools.mcp__codex_apps__github_fetch_commit({repo_full_name:repo,commit_sha:branch});
    if (r.isError) throw Error("Cannot read research branch");
    return r.structuredContent.commit.sha;
  };
  const protectedMain = await tools.mcp__codex_apps__github_fetch_commit({repo_full_name:repo,commit_sha:"main"});
  if(protectedMain.structuredContent?.commit?.sha !== m.protected_main) throw Error("Main moved; inspect before publication");
  let current = await head();
  if (current !== m.base && current !== state.remote_sha) throw Error("Remote changed; inspect before publication: " + current);
  if (current === m.base) {
    for (const f of m.files) {
      if (state.blobs.includes(f.sha)) continue;
      let content = "";
      for (let start=0; start<f.chars; start+=16000) {
        const code = "import subprocess,json; s=subprocess.check_output(['git','show'," + JSON.stringify(m.local_commit+":"+f.path) + "]).decode(); print(json.dumps(s["+start+":"+(start+16000)+"]))";
        content += JSON.parse(await shell("python -c " + quote(code),16000));
      }
      if (content.length !== f.chars) throw Error("Local content length mismatch: " + f.path);
      let r = await tools.mcp__codex_apps__github_create_blob({repository_full_name:repo,encoding:"utf-8",content});
      if (r.isError) r = await tools.mcp__codex_apps__github_create_blob({repository_full_name:repo,encoding:"utf-8",content});
      if (r.isError || r.structuredContent?.sha !== f.sha) throw Error("Blob mismatch: " + f.path);
      state.blobs.push(f.sha); await save(state); notify({uploaded:state.blobs.length,total:m.files.length});
    }
    for (const tree of m.trees) {
      let r = await tools.mcp__codex_apps__github_create_tree({repository_full_name:repo,tree_elements:tree.entries});
      if (r.isError) r = await tools.mcp__codex_apps__github_create_tree({repository_full_name:repo,tree_elements:tree.entries});
      if (r.isError || r.structuredContent?.sha !== tree.sha) throw Error("Tree mismatch: " + tree.path);
    }
    if (!state.remote_sha) {
      const r = await tools.mcp__codex_apps__github_create_commit({repository_full_name:repo,parent_sha:m.base,tree_sha:m.tree,message:m.message});
      if (r.isError || !r.structuredContent?.sha) throw Error("Commit creation failed");
      state.remote_sha=r.structuredContent.sha; await save(state);
    }
    const r = await tools.mcp__codex_apps__github_update_ref({repository_full_name:repo,branch_name:branch,expected_sha:m.base,sha:state.remote_sha,force:false});
    if (r.isError) throw Error("Ref update rejected; inspect remote head");
  }
  const verified=[];
  for (let i=0; i<m.files.length; i+=4) {
    const batch=await Promise.allSettled(m.files.slice(i,i+4).map(async f => {
      let r=await tools.mcp__codex_apps__github_fetch_file({repository_full_name:repo,path:f.path,ref:state.remote_sha});
      if (r.isError) r=await tools.mcp__codex_apps__github_fetch_file({repository_full_name:repo,path:f.path,ref:state.remote_sha});
      const d=r.structuredContent;
      if (r.isError || d?.sha!==f.sha || d?.content?.length!==f.chars) throw Error("Remote readback failed: "+f.path);
      return {path:f.path,sha:d.sha,verified:true};
    }));
    for (const r of batch) {if(r.status!=="fulfilled") throw r.reason;verified.push(r.value);}
  }
  if (await head() !== state.remote_sha) throw Error("Head changed during verification");
  const afterMain = await tools.mcp__codex_apps__github_fetch_commit({repo_full_name:repo,commit_sha:"main"});
  if(afterMain.structuredContent?.commit?.sha !== m.protected_main) throw Error("Main changed during publication");
  state.readback={head:state.remote_sha,tree:m.tree,files:verified}; await save(state);
  store("crossbehavior_remote_commit",state.remote_sha);store("crossbehavior_readback_complete",true);
  text({published:state.remote_sha,files:verified.length,remote_head_and_content_verified:true});
}
