"""Independent v7 Figure 3 audit, retaining the original numerical audits."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from matplotlib.colors import LinearSegmentedColormap,to_hex
import audit_figure_v6_common as base

BINDING_SHA='35d931855eed3646dcb91335c05ebc1edd7ac2f804eaf172be55e309191e2bb0'
NS={'s':'http://www.w3.org/2000/svg'}

def by_id(scene,id):
    found=[q for q in scene.iter() if q.get('id')==id]
    assert len(found)==1,'native group identity: '+id
    return found[0]

def signature(node):
    tag=node.tag.split('}')[-1]
    if tag in {'defs','text'}:return None
    return [tag,dict(sorted(node.attrib.items())),[value for child in node if (value:=signature(child)) is not None]]

def group_sha(node):
    return hashlib.sha256(json.dumps(signature(node),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def merged(rb,si):
    result=copy.deepcopy(rb)
    result['axes']+= [a for a in si['axes'] if not any((m.get('gid')or'').startswith('3b_correlations_') for m in a.get('meshes',[]))]
    return result

def visual(rb,journal):
    assert rb['figure']==3 and rb['journal']==journal
    base.close([rb['width_mm'],rb['height_mm']],[180,222],'approved v7 dimensions')
    assert [r['text'] for r in rb['labels']]==list('abcde'),'v7 panel inventory'
    assert all(r['fontsize']==8 and r['weight']=='bold' for r in rb['labels']),'v7 panel typography'
    assert {round(r['fontsize'],3) for r in rb['texts']}<={5,7,8},'v7 text typography'
    return dict(panels=list('abcde'),width_mm=180,height_mm=222,minimum_font_pt=5)

def records(root,rb,tables,scene):
    folder=root/'05_DATA/Analysis/figure_display/v7_figure3'
    saved=base.load(folder/'records.json')
    assert rb['records']==saved,'saved molecular record identity'
    payload=base.load(root/'05_DATA/Analysis/figure_display/v6_figure3/aminations_payload.json')
    ref=np.array(saved['reference']['pattern_points'],float)
    shared=next(r for r in payload['amine_rows'] if r['shared'])
    base.close(ref,shared['pattern_points'],'focal measured preference')
    order=sorted(range(95),key=lambda i:(not np.isfinite(ref[i]),-ref[i] if np.isfinite(ref[i]) else 0,i))
    assert order==saved['condition_permutation'],'one stable measured condition order'
    assert saved['sorted_condition_order']==[payload['condition_order'][i] for i in order]
    heat=LinearSegmentedColormap.from_list('independent_v7_preference',['#5A5A5A','#FFFFFF','#D9B748'])
    table=tables['V7_records_cells']
    assert len(table)==285,'all three 95-condition rows'
    for key,record in [('reference',saved['reference'])]+[(r['key'],r) for r in saved['records']]:
        values=np.array(record['pattern_points'],float)
        cells=[q for q in by_id(scene,'fingerprint_'+key) if q.get('data-original-index') is not None]
        assert len(cells)==95 and [int(q.get('data-original-index')) for q in cells]==order,'rendered one-order cells'
        source=table[table.record.eq(key)]
        assert source.position.tolist()==list(range(95)) and source.original_index.tolist()==order,'source one-order cells'
        base.close(source.preference_pp,values[order],'source record cell values')
        for pos,cell in enumerate(cells):
            value=values[order[pos]]
            assert cell.get('data-condition')==saved['sorted_condition_order'][pos]
            if np.isfinite(value):
                base.close([float(cell.get('data-value'))],[value],'native record cell')
                assert cell.get('fill')==to_hex(heat(np.clip((value+30)/60,0,1))),'native record colour'
            else:assert cell.get('data-value')=='missing','missing record cell'
        if key=='reference':continue
        role='amine' if key=='halide_record' else 'halide'
        rows=[r for r in payload[role+'_rows'] if not r['shared']]
        assert len(rows)==record['n_pairs']==(6 if role=='amine' else 9)
        assert record['member_ids']==[r['id'] for r in rows]
        assert all(saved['reference']['pair'] not in p for p in record['member_pairs'])
        array=np.array([r['pattern_points'] for r in rows],float)
        counts=np.isfinite(array).sum(axis=0)
        mean=np.divide(np.nansum(array,axis=0),counts,out=np.full(95,np.nan),where=counts>0)
        base.close(values,mean,'independent equal-weight finite molecular mean')
        assert record['contributor_counts']==counts.tolist()
        good=np.isfinite(ref)&np.isfinite(mean)
        base.close([record['r']],[np.corrcoef(ref[good],mean[good])[0,1]],'independent unbinned Pearson r')
        assert int(good.sum())==record['n_common']==93
        label=by_id(scene,key+'_correlation').text
        assert label==('r = 0.30' if key=='halide_record' else 'r = −0.10'),'rendered record r label'
        step=by_id(scene,'steps_'+key)
        assert [step.get(k) for k in ['data-ymin','data-ymax','data-axis-height-mm']]==['-13','13','16'],'shared step axes'
        segments=[q for q in step if q.get('data-segment') is not None]
        assert len(segments)==10
        y0=54 if key=='halide_record' else 80
        for j,positions in enumerate(np.array_split(np.arange(95),10)):
            segment=record['segments'][j];values_in_segment=values[np.array(order)[positions]]
            expected=np.nanmean(values_in_segment)
            assert segment['positions']==positions.tolist()
            base.close([segment['mean'],float(segments[j].get('data-mean'))],[expected,expected],'saved and native ten-segment steps')
            base.close([float(segments[j].get('y1')),float(segments[j].get('y2'))],[y0+8-expected/13*8]*2,'step plotted height')
            base.close([float(segments[j].get('x1')),float(segments[j].get('x2'))],[34+112*positions[0]/95,34+112*(positions[-1]+1)/95],'step common x positions')
    return dict(records=2,record_pairs=[6,9],conditions=95,common_conditions=93,cells=285,segments=20,tolerance=1e-9)

def native(root,scene,si_scene,rb,si_rb):
    folder=root/'05_DATA/Analysis/figure_display/v7_figure3'
    assert base.digest(folder/'BINDING.json')==BINDING_SHA,'approved source binding identity'
    binding=base.load(folder/'BINDING.json')
    for file,digest in binding['frozen_files'].items():assert base.digest(folder/file)==digest,file
    for file,digest in binding['scientific_inputs'].items():assert base.digest(root/'05_DATA'/file)==digest,file
    approved=ET.parse(folder/'Figure3_approved.svg').getroot();approved_si=ET.parse(folder/'FigS29_approved.svg').getroot()
    groups=base.load(folder/'native_display_groups.json')
    for id,digest in groups.items():
        drawn=scene if id in {'axes_10','axes_11'} else si_scene
        assert group_sha(by_id(drawn,id))==digest,'original quantitative native group: '+id
    for id in ['axes_4','axes_8']:
        assert ET.tostring(by_id(scene,id))==ET.tostring(by_id(approved,id)),'main correlation matrix: '+id
    # Complete SVG identity also covers hatches, labels, zero lines, structures and transforms.
    assert ET.tostring(scene)==ET.tostring(approved),'approved Figure3 full vector identity'
    assert ET.tostring(si_scene)==ET.tostring(approved_si),'approved FigS29 full vector identity'
    original=base.load(folder/'retained_artist_readback.json')
    assert rb['axes']==[dict(original['axes'][i],retained_source_axis_index=i,placement_transform_mm=([-27,53] if i==3 else [-36,53] if i==7 else [0,18])) for i in [3,7,9,10]],'main actual copied-artist identity'
    assert si_rb['axes']==[dict(original['axes'][i],retained_source_axis_index=i,placement_transform_mm=[0,-40]) for i in [1,2,3,4,5,6,7,8]],'SI actual copied-artist identity'
    assert si_rb['height_mm']==92 and si_rb['width_mm']==180 and si_rb['figure']==29
    combined=copy.deepcopy(si_scene)
    for child in scene:combined.append(copy.deepcopy(child))
    display=base.figure3_display(si_rb,combined)
    return dict(original_native_groups=len(groups),moved_heatmap_cells=17*95,full_vector_identity=True,D3_D11=display)

def retained_metadata(root,prov,rb):
    saved=base.load(root/'05_DATA/Analysis/figure_display/v7_figure3/retained_scientific_metadata.json')
    assert prov['panels']==saved['source_panels'],'archived scientific panel metadata'
    assert prov['panel_mapping']==rb['panel_mapping'],'current panel placement'
    assert prov['panel_metadata_scope']==saved['scope'],'archived panel label scope'
    return dict(original_panels=4,legacy_claim_contract=True,current_panel_mapping_separate=True)


def run():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);parser.add_argument('--qc-dir',type=Path)
    parser.add_argument('--artifact-dir',type=Path);parser.add_argument('--journal',choices=['science','natcatal'],default='science');parser.add_argument('--values-only',action='store_true')
    args=parser.parse_args();root=args.root.resolve();qc=(args.qc_dir or root/'08_REVIEW_AND_QC').resolve();qc.mkdir(parents=True,exist_ok=True)
    nature=root/'02_FIGURES/Journal_Formats/natcatal'
    artifact=args.artifact_dir.resolve() if args.artifact_dir else nature/'Figure3' if args.journal=='natcatal' else root/'04_SOURCE_DATA/Figure3'
    def output_path(relative):
        if args.artifact_dir:return artifact/Path(relative).name
        if args.journal=='natcatal':return nature/('Supplementary_Figures' if 'Supplementary_Figures' in relative else 'Main' if '/Main/' in relative else 'Figure3')/Path(relative).name
        return root/relative
    candidates=[artifact,artifact.parent/'figs29',nature/'Supplementary_SourceData' if args.journal=='natcatal' else root/'04_SOURCE_DATA/Supplementary_Figures']
    si_source=next((p for p in candidates if (p/'FigS29_readback.json').is_file()),None);assert si_source,'actual FigS29 readback missing'
    si_art=si_source if (si_source/'FigS29.svg').is_file() else nature/'Supplementary_Figures' if args.journal=='natcatal' else root/'02_FIGURES/Supplementary_Figures'
    rb=base.load(artifact/'Figure3_readback.json');si_rb=base.load(si_source/'FigS29_readback.json')
    tables=base.retained.read_stored_workbook(artifact/'Figure3_SourceData.xlsx');prov=base.load(artifact/'Figure3_SourceData_provenance.json')
    scene=ET.parse(output_path('02_FIGURES/Main/Figure3.svg')).getroot();si_scene=ET.parse(si_art/'FigS29.svg').getroot()
    checks={'retained_cells':base.retained_cells(3,root,tables),'independent_values':base.independent(3,root,tables,artifact),
        'artists':base.artist_values(3,root,tables,merged(rb,si_rb),artifact),'journal':visual(rb,args.journal),
        'records':records(root,rb,tables,scene),'native':native(root,scene,si_scene,rb,si_rb),
        'export':base.exported(3,output_path('02_FIGURES/Main/Figure3.pdf'),output_path('02_FIGURES/Main/Figure3.svg')),
        'SI_export':base.exported(29,si_art/'FigS29.pdf',si_art/'FigS29.svg'),
        'provenance':base.provenance(3,root,artifact,prov,output_path),
        'retained_scientific_metadata':retained_metadata(root,prov,rb)}
    faults=[]
    def reject(label,fn):
        try:fn()
        except (AssertionError,KeyError,StopIteration) as error:faults.append(dict(fault=label,rejected=True,reason=str(error)));return
        raise AssertionError('fault accepted: '+label)
    for mode in ['number','row']:
        bad={k:v.copy(deep=True) for k,v in tables.items()}
        if mode=='number':bad['A_correlations'].loc[0,'r']+=1
        else:bad['A_correlations']=bad['A_correlations'].iloc[:-1]
        reject('retained source '+mode,lambda:base.retained_cells(3,root,bad))
    for mode in ['old_layout','panel_font']:
        bad=copy.deepcopy(rb)
        if mode=='old_layout':bad['height_mm']=204;bad['labels']=bad['labels'][:4]
        else:bad['labels'][0]['fontsize']=6
        reject(mode,lambda:visual(bad,args.journal))
    for mode in ['order','member','r']:
        bad=copy.deepcopy(rb)
        if mode=='order':bad['records']['condition_permutation'].reverse()
        elif mode=='member':bad['records']['records'][0]['member_pairs'][0]=bad['records']['reference']['pair']
        else:bad['records']['records'][0]['r']+=.1
        reject('record '+mode,lambda:records(root,bad,tables,scene))
    for mode in ['cell','colour','step','r_label']:
        bad=copy.deepcopy(scene)
        if mode in ['cell','colour']:
            cell=next(q for q in by_id(bad,'fingerprint_halide_record') if q.get('data-value') not in [None,'missing'])
            cell.set('data-value',str(float(cell.get('data-value'))+1)) if mode=='cell' else cell.set('fill','#ff0000')
        elif mode=='step':next(q for q in by_id(bad,'steps_halide_record') if q.get('data-mean')).set('data-mean','99')
        else:by_id(bad,'halide_record_correlation').text='r = 0.99'
        reject('rendered record '+mode,lambda:records(root,rb,tables,bad))
    for mode in ['median','point']:
        bad=copy.deepcopy(rb)
        target=base.items(bad,'lines' if mode=='median' else 'collections')['3c_median_Buchwald-Hartwig|1_2' if mode=='median' else '3c_points_Buchwald-Hartwig|1_2']
        if mode=='median':target['x'][0]+=1
        else:target['offsets'][0][0]+=1
        reject('actual artist '+mode,lambda:base.artist_values(3,root,tables,merged(bad,si_rb),artifact))
    for id,is_si in [('axes_10',False),('axes_11',False),('3b_preference_amine',True)]:
        bad=copy.deepcopy(si_scene if is_si else scene);target=by_id(bad,id);target.remove(list(target)[-1])
        reject('native group '+id,lambda:native(root,scene if is_si else bad,bad if is_si else si_scene,rb,si_rb))
    bad=copy.deepcopy(prov);next(iter(bad['inputs']['files'].values()))['sha256']='0'*64
    reject('input provenance',lambda:base.provenance(3,root,artifact,bad,output_path))
    bad=copy.deepcopy(prov);bad['panels']['A']['noise_calibration']['duplicate_agreement']+=.1
    reject('retained scientific metadata',lambda:retained_metadata(root,bad,rb))
    if not args.values_only:
        with tempfile.TemporaryDirectory(prefix='v7_figure3_',dir=qc) as folder:
            command=[sys.executable,'-B',str(root/'06_CODE/Figures/figure3.py'),'--out',folder,'--journal',args.journal]
            completed=subprocess.run(command,cwd=root,capture_output=True,text=True,encoding='utf-8',errors='replace')
            assert completed.returncode==0,completed.stdout+completed.stderr
            assert base.load(Path(folder)/'Figure3_SourceData_provenance.json')==prov,'isolated provenance differs'
            for row in prov['outputs'].values():assert base.digest(Path(folder)/Path(row['file']).name)==row['sha256'],'isolated output: '+row['file']
        checks['fresh_rebuild']=dict(result='PASS',outputs=len(prov['outputs']),deterministic=True)
    result=dict(result='PASS',figure=3,journal=args.journal,checks=checks,fault_injections=faults)
    destination=qc/'Figure3_v6_audit.json';destination.write_text(json.dumps(result,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(result='PASS',figure=3,journal=args.journal,faults_rejected=len(faults),report=str(destination)),indent=2))

if __name__=='__main__':run()
