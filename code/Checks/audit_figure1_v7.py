"""Independent audit of v7 Figure 1 evidence and the v8 display amendment."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,re,subprocess,sys,tempfile
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pymupdf
from rdkit import Chem
from reportlab.pdfbase import pdfmetrics

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'06_CODE/Figures'))
from structure_art import load as load_art
from v6_figure1 import fonts

NS={'s':'http://www.w3.org/2000/svg'};PT=72/25.4
RECORD_SHA='515e2813b4d18c6b93177480f9192cdbeb08c1929ec2bf2f9511433916e9697c'
DISPLAY='05_DATA/Analysis/figure_display/v7_figure1'
AMENDMENT=DISPLAY+'/Figure1_DISPLAY_AMENDMENT_v8.json'
AMENDMENT_SHA='cb361f14841d11fc567b439414ba1822da548a40edbfdd48e6cd353427abb382'
BLUE,ORANGE,DARK='#0072B2','#D55E00','#4D4D4D'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def resolve(root,rel):
 bases=[root]+([root.parent.parent]if root.parent.name=='_rebuild'else[])
 aliases=[('05_DATA/Analysis/figure_display/','figure_inputs/'),('02_FIGURES/ChemDraw/','figures/structures/'),('06_CODE/','code/'),('04_SOURCE_DATA/','source_data/')]
 candidates=[root/rel]
 for b in bases:
  for old,new in aliases:
   if rel.startswith(old):candidates.append(b/(new+rel[len(old):]))
 p=next((p for p in candidates if p.is_file()),None)
 assert p is not None,('missing portable input',rel)
 return p
def one(svg,id):
 a=svg.findall(f".//*[@id='{id}']");assert len(a)==1,('unique artist',id,len(a));return a[0]
def box_distance(a,b):return math.hypot(max(a[0]-b[2],b[0]-a[2],0),max(a[1]-b[3],b[1]-a[3],0))
def text_box(n):
 size=float(n.get('font-size'))*PT;font='Arial-Bold'if n.get('font-weight')=='bold'else'Arial'
 w=pdfmetrics.stringWidth(n.text or'',font,size)/PT;x,y=float(n.get('x')),float(n.get('y'))
 x-=w*{'start':0,'middle':.5,'end':1}[n.get('text-anchor','start')];a,d=pdfmetrics.getAscentDescent(font,size)
 return[x,y-a/PT,x+w,y-d/PT]

def discovery_reading_order(svg):
 ids=['median_ownership','role_switch_title','halide_discovery','prediction_discovery']
 nodes=[one(svg,id)for id in ids]
 ys=[float(n.get('y'))for n in nodes]
 assert all(a<b for a,b in zip(ys,ys[1:])),('discovery vertical reading order',ys)
 actual=list(one(svg,'03_discoveries').iter())
 indices=[actual.index(n)for n in nodes]
 assert indices==sorted(indices),('discovery document reading order',indices)
 return ids

def verify_display_amendment(svg,records,root):
 amendment=read(resolve(root,AMENDMENT));assert sha(resolve(root,AMENDMENT))==AMENDMENT_SHA
 assert amendment['source_records_sha256']==RECORD_SHA and amendment['original_rule_sha256']==records['rule']['sha256']
 assert svg.find('s:title',NS).text==amendment['title'],'v8 display title'
 meta=read_metadata(svg)
 assert meta['display_revision']==amendment['revision'] and meta['display_amendment_sha256']==AMENDMENT_SHA
 assert meta['display_title']==amendment['title']
 assert meta['primary_discovery_numbers']==['80%','10/10','≈3×']==amendment['primary_discovery_numbers']
 for entry in amendment['right_column_headlines']:assert one(svg,entry['id']).text==entry['text']
 return dict(revision=amendment['revision'],title=amendment['title'],reading_order=discovery_reading_order(svg),amendment_sha256=AMENDMENT_SHA)

def local_prior_work(svg):
 targets={'guide_leader':('cite_ligand_guides',[29,18,37,21]),'general_leader':('cite_general_conditions',[25,28,41,31]),
          'informer_leader':('cite_informer',[4,60,24,74]),'robustness_leader':('cite_robustness',[30,86,34,90])}
 gaps=[]
 for lead,(label,target)in targets.items():
  n=one(svg,lead);p=[[float(v)for v in pair.split(',')]for pair in n.get('points').split()]
  assert 2<=len(p)<=3
  length=sum(math.dist(a,b)for a,b in zip(p,p[1:]));assert length<=9,('long prior-work leader',lead,length)
  # The attached label includes its superscript citation, which must also stay clear.
  boxes=[text_box(one(svg,id))for id in [label,label+'_citation']]
  combined=[min(b[0]for b in boxes),min(b[1]for b in boxes),max(b[2]for b in boxes),max(b[3]for b in boxes)]
  gap=box_distance([*p[0],*p[0]],combined)
  assert 2/PT<=gap<=3,('prior-work label clearance or attachment',label,gap)
  assert box_distance([*p[-1],*p[-1]],target)<=1,('unattached prior-work leader',lead)
  gaps.append(dict(label=label,leader_length_mm=length,label_gap_mm=gap))
 assert not svg.findall(".//*[@id='cite_recommendation']")
 assert not any('condition recommendation'in(n.text or'').lower()for n in svg.findall('.//s:text',NS))
 return gaps

def verify_svg(svg,records,root):
 fonts()
 assert svg.get('width')=='180mm'and svg.get('height')=='130mm'
 assert not svg.findall('.//s:image',NS)
 meta=read_metadata(svg)
 assert meta['version']=='v7_formal'and meta['source_records_sha256']==RECORD_SHA
 assert meta['scientific_values_recalculated']is False and meta['journal_invariant']is True
 display=verify_display_amendment(svg,records,root)
 assert meta['owner_blocks']==records['owner_blocks']and meta['transfer']==records['transfer']
 ids=[n.get('id')for n in svg.iter()if n.get('id')];assert len(ids)==len(set(ids))
 actual_texts=svg.findall('.//s:text',NS)
 # Each recorded artist must match an actual text node; coordinates are read from the artwork.
 for n in actual_texts:
  role=n.get('data-role');size=round(float(n.get('font-size'))*PT,6)
  assert n.get('font-family')=='Arial'and n.get('font-style')=='normal'
  assert size in((7,5.25)if role=='atom'else(8,)if role=='panel'else(5,7)),(n.get('id'),size)
  if role not in('atom','citation'):assert len((n.text or'').split())<=5
  b=text_box(n);assert 0<=b[0]<b[2]<=180 and 0<=b[1]<b[3]<=130
  matches=[t for t in meta['texts']if t['id']==n.get('id')and t['text']==(n.text or'')and t['role']==role and abs(t['fontsize']-size)<1e-5 and np.allclose(t['bbox_mm'],b,rtol=0,atol=1e-7)]
  assert matches,('unbound or changed actual text',n.get('id'),n.text)
 assert len(actual_texts)==len(meta['texts'])
 for id,value in [('reaction_count','177,039 reactions'),('scope','15 screens · 14 transformations'),('outside_pairs','844 outside-screen pairs'),('external_screens','2 external public screens'),('recurrence_63pct','63% recur with partners'),('knowledge_gap','Which molecule carries the conditions?'),('halide_discovery','10/10 halide screens'),('median_ownership','Median 80% to one molecule'),('median_denominator','Owner / molecular component'),('prediction_discovery','≈3× condition-profile prediction')]:assert one(svg,id).text==value
 assert [one(svg,'panel_'+c).text for c in 'abc']==list('abc')
 assert one(svg,'owner_source').text=='Order: Fig. 2a'and one(svg,'transfer_source').text=='Set A · Fig. 5a'
 blocks=svg.findall(".//*[@data-role='owner_identity']");assert len(blocks)==15
 assert [x.get('fill')for x in blocks]==[BLUE]*10+[ORANGE,DARK,DARK,DARK,ORANGE]
 assert len({(x.get('y'),x.get('width'),x.get('height'))for x in blocks})==1,'owner bars must be equal identity blocks'
 assert all(float(x.get('width'))==2.75 and float(x.get('height'))==3.1 for x in blocks)
 assert [x.get('data-screen')for x in blocks]==[x['screen']for x in records['owner_blocks']]
 assert all(x.get('data-value')is None and x.get('data-owner-share')is None for x in blocks)
 bars=svg.findall(".//*[@data-role='prediction_bar']");assert len(bars)==2
 for side in ['owner','partner']:
  n=one(svg,'transfer_'+side);r=next(x for x in records['transfer']['rows']if x['side']==side)
  assert float(n.get('data-value-percent'))==r['value']and abs(float(n.get('width'))-r['value']*1.02)<1e-12
  assert float(n.get('x'))==143 and float(n.get('height'))==.9
 ctext=' '.join(n.text or''for n in one(svg,'03_discoveries').findall('.//s:text',NS))
 assert not any(x in ctext for x in ['22.7','7.5','12.7','3.8','Set B','Owner share','239.9'])
 assert not re.search(r'\bR\b',' '.join(n.text or''for n in actual_texts))
 groups=svg.findall('.//*[@data-master]');assert len(groups)==6
 expected_keys={'recurring_aryl_bromide':'recurring_halide','recurring_amine1':'recurring_amine1','recurring_amine2':'recurring_amine2','recurring_amine3':'recurring_amine3','role_switch_owner':'role_switch','role_switch_partner':'role_switch'}
 geometry=[]
 for id,key in expected_keys.items():
  n=one(svg,id);entry=next(x for x in records['vectors']['items']if x['name']==key);rel=f'02_FIGURES/ChemDraw/v7/f1/{key}.svg'
  assert n.get('data-master')==rel and n.get('data-smiles')==entry['smiles']
  source=resolve(root,rel);master=resolve(root,rel.replace('.svg','.cdxml'))
  assert sha(source)==entry['svg_sha256']and sha(master)==entry['cdxml_sha256']
  art=load_art(source);k=7/art['label_px']/PT;assert float(n.get('data-uniform-scale'))==k
  polygons=n.findall('s:polygon',NS);assert len(polygons)==len(art['polygons'])and len(polygons)>0
  coords=lambda p:np.asarray([[float(v)for v in s.split(',')]for s in p.get('points').split()])
  shift=coords(polygons[0])[0]-art['polygons'][0][0]*k
  for p,original in zip(polygons,art['polygons']):
   actual,expected=coords(p),original*k+shift
   assert actual.shape==expected.shape,('native polygon point count changed',id)
   assert np.allclose(actual,expected,rtol=0,atol=1e-8)
  labels=n.findall('s:text',NS);assert len(labels)==len(art['labels'])
  for label,original in zip(labels,art['labels']):
   assert label.text==original['text']and abs(float(label.get('font-size'))*PT-7*original['scale'])<1e-8
   assert np.allclose([float(label.get('x')),float(label.get('y'))],[original['x']*k+shift[0],original['y']*k+shift[1]],atol=1e-8,rtol=0)
  xml=ET.parse(master).getroot();assert float(xml.get('BondLength'))==14.4 and float(xml.get('LabelSize'))==10
  mols=Chem.MolsFromCDXML(master.read_text(encoding='utf-8'));assert len(mols)==1
  assert Chem.MolToSmiles(mols[0])==Chem.MolToSmiles(Chem.MolFromSmiles(entry['smiles']))
  atoms={a.get('id'):np.asarray(list(map(float,a.get('p').split())))for a in xml.iter('n')if a.get('p')};adj={a:[]for a in atoms}
  for bond in xml.iter('b'):
   a,b=bond.get('B'),bond.get('E')
   if a in atoms and b in atoms:adj[a].append(b);adj[b].append(a)
  angles=[]
  for a,neighbors in adj.items():
   if len(neighbors)!=3:continue
   for i in range(3):
    for j in range(i):
     u,v=atoms[neighbors[i]]-atoms[a],atoms[neighbors[j]]-atoms[a]
     angles.append(math.degrees(math.acos(np.clip(float(u@v/np.linalg.norm(u)/np.linalg.norm(v)),-1,1))))
  dev=max([abs(a-120)for a in angles],default=0);assert dev<5,(key,dev)
  geometry.append(dict(id=id,identity=entry['smiles'],native_vector_preserved=True,max_sp2_deviation_deg=dev))
 prior=local_prior_work(svg)
 for key,number in records['citations'].items():assert one(svg,key+'_citation').text==number
 return dict(metadata=meta,display_amendment=display,native_D20_geometry=geometry,local_prior_work=prior,owner_blocks=15,prediction_bars=2)

def read_metadata(svg):
 n=svg.find('s:metadata',NS);assert n is not None;return json.loads(n.text)

def replay_identity_rule(root,records):
 selection=read(resolve(root,records['selection']['path']))
 assert selection['outcome_fields_used']==[]and selection['read_columns']==['screen','sub_1_smiles','sub_2_smiles']
 roots=[root]+([root.parent.parent]if root.parent.name=='_rebuild'else[])
 source=next((p for b in roots for p in [b/'10_PUBLIC_RELEASE_CANDIDATE/data/measurements_A1_analysis_arrays.csv.gz',b/'data/measurements_A1_analysis_arrays.csv.gz']if p.is_file()),None)
 assert source is not None,'identity-only released source missing'
 assert sha(source)==selection['source_sha256']
 d=pd.read_csv(source,usecols=selection['read_columns'],dtype=str);d=d[d.screen.isin(selection['screens'])]
 canon=lambda s:Chem.MolToSmiles(Chem.MolFromSmiles(s))
 pairs={(canon(a),canon(b))for a,b in zip(d.sub_1_smiles,d.sub_2_smiles)}
 # Rebuild the complete identity ranking independently of the saved candidates.
 bromide=Chem.MolFromSmarts('[Br]-[a]');amine=Chem.MolFromSmarts('[NX3;H1,H2;+0;!$(N-[C,S,P]=[O,S,N])]')
 molecules={s:Chem.MolFromSmiles(s)for pair in pairs for s in pair}
 neutral=lambda m:Chem.GetFormalCharge(m)==0 and len(Chem.GetMolFrags(m))==1
 halogen_count=lambda m:sum(a.GetSymbol()in('F','Cl','Br','I')for a in m.GetAtoms())
 eligible_amines={s for s,m in molecules.items()if neutral(m)and m.GetNumHeavyAtoms()<=10 and len(m.GetRingInfo().AtomRings())<=1 and m.HasSubstructMatch(amine)and halogen_count(m)==0}
 eligible_bromides={s for s,m in molecules.items()if neutral(m)and m.GetNumHeavyAtoms()<=18 and len(m.GetRingInfo().AtomRings())<=2 and len(m.GetSubstructMatches(bromide))==1 and halogen_count(m)==1 and any(len(r)==6 and all(m.GetAtomWithIdx(i).GetIsAromatic()for i in r)for r in m.GetRingInfo().AtomRings())}
 ranks=[]
 for bromide_identity in eligible_bromides:
  partners=sorted({a for a,b in pairs if b==bromide_identity and a in eligible_amines})
  if len(partners)>=3:ranks.append(dict(smiles=bromide_identity,eligible_amine_partners=len(partners),heavy_atoms=molecules[bromide_identity].GetNumHeavyAtoms(),partners=partners))
 ranks.sort(key=lambda x:(-x['eligible_amine_partners'],x['heavy_atoms'],x['smiles']))
 assert ranks==selection['halide_rank'],'saved bromide ranking differs from released identities'
 amine_ranks=[dict(smiles=a,recorded_halide_partners=len({b for x,b in pairs if x==a}),heavy_atoms=molecules[a].GetNumHeavyAtoms())for a in ranks[0]['partners']]
 amine_ranks.sort(key=lambda x:(-x['recorded_halide_partners'],x['heavy_atoms'],x['smiles']))
 assert amine_ranks==selection['amine_rank'],'saved amine ranking differs from released identities'
 h=selection['selected_halide'];amines=selection['selected_amines'];assert len(amines)==3 and len(set(amines))==3
 assert all((a,h)in pairs for a in amines)
 assert selection['selected_halide']==selection['halide_rank'][0]['smiles']
 assert selection['selected_amines']==[x['smiles']for x in selection['amine_rank'][:3]]
 assert selection['halide_rank']==sorted(selection['halide_rank'],key=lambda x:(-x['eligible_amine_partners'],x['heavy_atoms'],x['smiles']))
 assert selection['amine_rank']==sorted(selection['amine_rank'],key=lambda x:(-x['recorded_halide_partners'],x['heavy_atoms'],x['smiles']))
 for entry in selection['amine_rank']:
  assert entry['recorded_halide_partners']==len({b for a,b in pairs if a==entry['smiles']})
 assert sha(resolve(root,records['rule']['path']))==selection['rule_sha256']==records['rule']['sha256']
 return dict(outcome_fields_used=[],selected_pairs=selection['selected_pairs'],identity_connections_verified=3,rule_sha256=selection['rule_sha256'])

def controls(svg,records,root):
 results=[]
 def reject(name,mutate,check=None):
  clone=copy.deepcopy(svg);mutate(clone)
  try:(check or(lambda s:verify_svg(s,records,root)))(clone)
  except (AssertionError,KeyError):results.append(dict(test=name,rejected=True))
  else:raise AssertionError('accepted negative control: '+name)
 reject('wrong_owner_colour',lambda s:one(s,'owner_role_01').set('fill',ORANGE))
 def remove_block(s):
  n=one(s,'owner_role_15');next(p for p in s.iter()if n in list(p)).remove(n)
 reject('missing_fifteenth_owner',remove_block)
 reject('changed_owner_order',lambda s:one(s,'owner_role_01').set('data-screen','wrong'))
 reject('numeric_owner_bars_restored',lambda s:one(s,'owner_role_01').set('height','8.2'))
 reject('wrong_prediction_bar_value',lambda s:one(s,'transfer_partner').set('width','20'))
 reject('changed_principal_result',lambda s:setattr(one(s,'prediction_discovery'),'text','≈9× condition-profile prediction'))
 reject('halide_first_discovery_order',lambda s:one(s,'halide_discovery').set('y','17'),discovery_reading_order)
 reject('old_figure_title',lambda s:setattr(s.find('s:title',NS),'text',records['title']),lambda s:verify_display_amendment(s,records,root))
 def extra_value(s):
  t=copy.deepcopy(one(s,'prediction_discovery'));t.set('id','extra_numeric_value');t.text='22.7';one(s,'03_discoveries').append(t)
 reject('extra_discovery_numbers',extra_value)
 reject('floating_prior_work_label',lambda s:one(s,'cite_informer').set('x','120'),local_prior_work)
 reject('leader_crosses_citation',lambda s:one(s,'general_leader').set('points','20.4,30.4 25.5,29.4'),local_prior_work)
 reject('long_crossing_prior_work_leader',lambda s:one(s,'robustness_leader').set('points','4,86 150,10'),local_prior_work)
 def generic_r(s):
  g=one(s,'recurring_aryl_bromide')
  for child in list(g):g.remove(child)
  t=ET.SubElement(g,'{'+NS['s']+'}text',{'x':'14','y':'67','font-family':'Arial','font-size':str(7/PT),'font-style':'normal','data-role':'annotation'});t.text='R'
 reject('real_halide_replaced_by_R_icon',generic_r)
 reject('native_bond_geometry_changed',lambda s:one(s,'recurring_amine3').find('s:polygon',NS).set('points','1,1 2,2 3,3'))
 reject('undersized_font',lambda s:one(s,'knowledge_gap').set('font-size','1'))
 return results

def audit(root,art,data):
 records=read(resolve(root,DISPLAY+'/records.json'));assert sha(resolve(root,DISPLAY+'/records.json'))==RECORD_SHA
 svg=ET.parse(art/'Figure1.svg').getroot();report=verify_svg(svg,records,root);meta=report.pop('metadata')
 actual=read(data/'Figure1_v7_readback.json');assert {k:v for k,v in actual.items()if k not in('pdf_fonts','pdf_raster_images')}==meta
 binding=read(data/'Figure1_display_bindings.json');assert binding['records']==records and binding['records_sha256']==RECORD_SHA
 assert binding['display_amendment']==read(resolve(root,AMENDMENT)) and binding['display_amendment_sha256']==AMENDMENT_SHA
 assert (data/'Figure1_DISPLAY_RULE.md').read_bytes()==resolve(root,records['rule']['path']).read_bytes()
 assert (data/'Figure1_DISPLAY_AMENDMENT_v8.json').read_bytes()==resolve(root,AMENDMENT).read_bytes()
 report['identity_rule']=replay_identity_rule(root,records)
 with pymupdf.open(art/'Figure1.pdf')as pdf:
  assert len(pdf)==1;p=pdf[0];assert np.allclose([p.rect.width/PT,p.rect.height/PT],[180,130],atol=1e-5)
  spans=[s for b in p.get_text('dict')['blocks']for l in b.get('lines',[])for s in l['spans']]
  assert all('Arial'in s['font']and s['size']>=4.999 for s in spans)
  assert all(p.rect.contains(pymupdf.Rect(s['bbox']))for s in spans)
  fs=[f for f in p.get_fonts()if f[2]=='TrueType'];assert len(fs)==2
  assert all(len(pdf.extract_font(f[0])[3])>1000 for f in fs)
  assert not p.get_images(full=True)
  assert hashlib.sha256(p.get_pixmap(dpi=600,alpha=False).tobytes('png')).hexdigest()==sha(art/'Figure1.png')
  report['export']=dict(embedded_Arial=True,fonts=[f[3]for f in fs],raster_images=0,png_matches_pdf=True,text_in_page=True)
 # Legacy scientific cells and payload remain fully auditable, independent of the new layout.
 import audit_figure1 as legacy
 saved=read(resolve(root,'05_DATA/Analysis/figure_display/v6_figure1/records.json'))
 legacy.workbook_check(data/'Figure1_SourceData.xlsx',saved)
 payload=read(data/'figure1_payload.json');assert payload['schema']=='v6_frozen_histories'
 legacy.equal_saved(payload['records'],saved['records']);legacy.equal_saved(payload['reference'],saved['reference'])
 for entry in records['legacy']:assert sha(data/entry['file'])==entry['sha256']==sha(resolve(root,entry['path']))
 report['legacy_evidence']=dict(byte_identical=True,display_cells=285,saved_steps=20,contributor_rows=190,member_pairs=15)
 provs=list(data.glob('*_provenance.json'));assert len(provs)==1 and provs[0].name=='Figure1_SourceData_provenance.json'
 prov=read(provs[0]);assert prov['generated_by']=='06_CODE/Figures/figure1.py'and prov['version']=='v7_formal'
 assert prov['outputs']['xlsx']['file']=='04_SOURCE_DATA/Figure1/Figure1_SourceData.xlsx'
 for group in prov['inputs'].values():
  for entry in group.values():assert sha(resolve(root,entry['file']))==entry['sha256']
 for entry in prov['outputs'].values():
  p=(art if entry['file'].startswith('02_FIGURES/Main/')else data)/Path(entry['file']).name
  assert sha(p)==entry['sha256'],entry['file']
 report['negative_controls']=controls(svg,records,root)
 report.update(status='PASS',result='PASS',figure=1,version='v7_formal',records_sha256=RECORD_SHA,
               checks={k:dict(result='PASS')for k in ['actual_svg','v8_display_amendment','frozen_values','identity_rule','native_geometry','fonts_vectors','legacy_evidence','provenance','fault_injections']},
               fault_injections=report['negative_controls'],scientific_values_recalculated=False)
 return report

def main():
 parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--qc-dir',type=Path)
 parser.add_argument('--out',type=Path);parser.add_argument('--artifact-dir',type=Path);parser.add_argument('--journal',choices=['science','natcatal'],default='science');parser.add_argument('--values-only',action='store_true')
 args=parser.parse_args();root=args.root.resolve();artifact=args.artifact_dir or args.out
 if artifact:art=data=artifact.resolve()
 else:
  art=root/('02_FIGURES/Main'if args.journal=='science'else'02_FIGURES/Journal_Formats/natcatal/Main')
  data=root/('04_SOURCE_DATA/Figure1'if args.journal=='science'else'02_FIGURES/Journal_Formats/natcatal/Figure1')
 qc=args.qc_dir or root/'08_REVIEW_AND_QC/v8_title_20261028/figure1';qc.mkdir(parents=True,exist_ok=True)
 report=audit(root,art,data)
 if not args.values_only:
  with tempfile.TemporaryDirectory(prefix='figure1_v7_',dir=qc)as temp:
   p=subprocess.run([sys.executable,'-B',str(root/'06_CODE/Figures/figure1.py'),'--out',temp,'--data-root',str(root),'--journal',args.journal],capture_output=True,text=True,encoding='utf-8',errors='replace');assert p.returncode==0,p.stdout+p.stderr
   prov=read(data/'Figure1_SourceData_provenance.json')
   for entry in prov['outputs'].values():assert sha(Path(temp)/Path(entry['file']).name)==entry['sha256']
   assert read(Path(temp)/'Figure1_SourceData_provenance.json')==prov
  report['checks']['fresh_rebuild']=dict(result='PASS',deterministic=True)
 for name in ['Figure1_v7_audit.json','Figure1_v6_audit.json']:(qc/name).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps({k:report[k]for k in ['status','result','version','checks','negative_controls']},indent=2))

if __name__=='__main__':main()
