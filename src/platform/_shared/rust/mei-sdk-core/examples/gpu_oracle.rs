//! Diagnostic numeric reference for WebGPU. No Agent execution or publication.
use mei_sdk_core::{
    model::{Arch, NeedleModel},
    packed::PackedWeights,
};
use serde_json::{json, Value};
fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let path = std::path::Path::new(&args[1]);
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(path.join("mei-model.json")).unwrap()).unwrap();
    let weights = PackedWeights::parse(std::fs::read(path.join("tensors.bin")).unwrap()).unwrap();
    let model = NeedleModel::without_full_f32_cache(Arch::from_manifest(&manifest), weights);
    let mut cache = None;
    let first = model
        .forward_bounded_last_logits(&[2], &mut cache, false, None, 0)
        .unwrap();
    let a = first.last_logits().unwrap().to_vec();
    cache = Some(first.cache);
    let second = model
        .forward_bounded_last_logits(&[37], &mut cache, false, Some(&[2]), 0)
        .unwrap();
    let b = second.last_logits().unwrap().to_vec();
    let many = model
        .forward_bounded_last_logits(&[2, 37], &mut None, false, None, 0)
        .unwrap();
    let ids=(0..128u32).map(|i| if i==0 {2} else {(i*137+37)%24000}).collect::<Vec<_>>();
    let mut cc=None;
    let pref=model.forward_bounded_last_logits(&ids,&mut cc,false,None,0).unwrap();
    let pf=pref.last_logits().unwrap().to_vec();cc=Some(pref.cache);
    let next=model.forward_bounded_last_logits(&[37],&mut cc,false,Some(&ids),0).unwrap();
    println!(
        "{}",
        json!({"kind":"native-cq2-numeric-oracle","first_logits":a,"second_logits":b,"two_token_logits":many.last_logits().unwrap(),"varied128_logits":pf,"next128_logits":next.last_logits().unwrap()})
    );
}
