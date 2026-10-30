"""Portable exports of the authorized v7 Figure 3 and displaced barcode figure.

Approved vector display assets are bound to the frozen scientific arrays. No
correlation, residual, null distribution, eligibility or selection is refitted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from io import BytesIO
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pymupdf
from matplotlib import font_manager
from reportlab.graphics import renderPDF
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from svglib.svglib import svg2rlg
from svglib.fonts import register_font_family

ROOT = Path(__file__).resolve().parents[2]
NS = {'s':'http://www.w3.org/2000/svg'}
MM = 25.4 / 72
FONT_SHA = {'Arial':'c9b76220a5be42ead4733611e417cd65c5fd8aeaa33eb56576ac378a37d130a1',
            'Arial-Bold':'766f06ac8761f82f25d032a220e89438f6064591af9915061f20b949efdedf69'}
BINDING_SHA256='8c45a19589b2338ead55c022b69df3da974ff8039bfcfddf075fc856a35610fe'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path,value):
    from figure_export import clean
    Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def by_id(root,id):
    return next(q for q in root.iter() if q.get('id')==id)

def fonts():
    paths={}
    for name,digest in FONT_SHA.items():
        candidates=[Path(f.fname) for f in font_manager.fontManager.ttflist if f.name=='Arial']
        path=next((p for p in candidates if p.is_file() and sha(p)==digest),None)
        if path is None:raise FileNotFoundError(f'Pinned Arial face required: {name}, SHA-256 {digest}')
        paths[name]=str(path)
        if name not in pdfmetrics.getRegisteredFontNames():pdfmetrics.registerFont(TTFont(name,str(path)))
    register_font_family('Arial',paths['Arial'],bold=paths['Arial-Bold'])
    return paths

def check_assets(data):
    here=data/'Analysis/figure_display/v7_figure3'
    assert sha(here/'BINDING.json')==BINDING_SHA256,'approved v7 display binding'
    binding=load(here/'BINDING.json')
    for name,digest in binding['frozen_files'].items():assert sha(here/name)==digest, name
    for relative,digest in binding['scientific_inputs'].items():assert sha(data/relative)==digest,relative
    main=ET.parse(here/'Figure3_approved.svg').getroot()
    si=ET.parse(here/'FigS29_approved.svg').getroot()
    rec=load(here/'records.json')
    assert [r['n_pairs'] for r in rec['records']]==[6,9]
    assert [r['r'] for r in rec['records']]==[0.3025779035389302,-0.10458708553363207]
    assert all(rec['reference']['pair'] not in r['member_pairs'] for r in rec['records'])
    for key,values in [('reference',rec['reference']['pattern_points'])]+[(r['key'],r['pattern_points']) for r in rec['records']]:
        group=by_id(main,'fingerprint_'+key)
        cells=[q for q in group if q.get('data-original-index') is not None]
        assert len(cells)==95,key
        assert [int(q.get('data-original-index')) for q in cells]==rec['condition_permutation']
        for cell in cells:
            value=values[int(cell.get('data-original-index'))]
            drawn=cell.get('data-value')
            assert drawn==('missing' if value is None else str(value)),(key,drawn,value)
    for id,digest in binding['approved_v6_native_groups_sha256'].items():
        scene=main if id in {'axes_4','axes_8'} else si
        assert hashlib.sha256(ET.tostring(by_id(scene,id))).hexdigest()==digest,id
    for scene in [main,si]:
        ids=[q.get('id') for q in scene.iter() if q.get('id')]
        assert len(ids)==len(set(ids))
        for q in scene.iter():
            for value in q.attrib.values():
                assert all(id in ids for id in re.findall(r'url\(#([^)]+)\)',value))
            link=q.get('{http://www.w3.org/1999/xlink}href','')
            assert not link.startswith('#') or link[1:] in ids
    return here,binding,rec

def write_workbook(sheets,path):
    with pd.ExcelWriter(path,engine='openpyxl') as writer:
        for name,frame in sheets.items():
            frame.to_excel(writer,sheet_name=name,index=False)
            sheet=writer.sheets[name];sheet.freeze_panes='A2'
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width=min(60,max(12,max(len(str(c.value or '')) for c in column)+2))
    with ZipFile(path) as archive:items=[(i,archive.read(i.filename)) for i in archive.infolist()]
    result=BytesIO()
    with ZipFile(result,'w',ZIP_DEFLATED) as archive:
        for info,data in items:
            if info.filename=='docProps/core.xml':data=re.sub(rb'<dcterms:(?:created|modified)[^>]*>[^<]*</dcterms:(?:created|modified)>',b'',data)
            entry=ZipInfo(info.filename,date_time=(1980,1,1,0,0,0));entry.compress_type=ZIP_DEFLATED;entry.external_attr=info.external_attr
            archive.writestr(entry,data)
    Path(path).write_bytes(result.getvalue())

def tables(data,rec):
    p=data/'Analysis/figure_display/v6_figure3'
    payload=load(p/'aminations_payload.json')
    cells=[];steps=[];members=[]
    for key,row in [('reference',rec['reference'])]+[(r['key'],r) for r in rec['records']]:
        for pos,index in enumerate(rec['condition_permutation']):
            cells.append(dict(record=key,position=pos,condition=rec['sorted_condition_order'][pos],
                              original_index=index,preference_pp=row['pattern_points'][index],
                              contributing_pairs=(row.get('contributor_counts') or [1]*95)[index]))
        for index,segment in enumerate(row.get('segments',[])):
            steps.append(dict(record=key,segment=index,positions=','.join(map(str,segment['positions'])),
                              mean_preference_pp=segment['mean'],finite_conditions=segment['finite_conditions']))
        for pair in row.get('member_pairs',[]):members.append(dict(record=key,pair=pair))
    bars=[]
    for role in ['amine','halide']:
        for row in payload[role+'_rows']:
            for condition,value in zip(payload['condition_order'],row['pattern_points']):
                bars.append(dict(replaced_role=role,molecule_id=row['id'],pair=row.get('pair'),condition=condition,preference_pp=value))
    result=pd.read_excel(data/'Analysis/figure_display/v5_reference_tables/preference.xlsx',sheet_name=None,dtype=object,keep_default_na=False,na_values=[''])
    def rewrite(value):
        if not isinstance(value,str):return value
        for n,key in {1:'overview',2:'preference',3:'screen_shares',4:'molecule_context',5:'transfer'}.items():
            value=value.replace(f'04_SOURCE_DATA/Figure{n}/Figure{n}_SourceData.xlsx',f'05_DATA/Analysis/figure_display/v5_reference_tables/{key}.xlsx')
        return value.replace('04_SOURCE_DATA/Figure2/figure2_payload.json','05_DATA/Analysis/figure_display/v5_reference_tables/preference_payload.json')
    for frame in result.values():
        for column in frame:
            if frame[column].dtype==object:frame[column]=frame[column].map(rewrite)
    result.update({'V7_README':pd.DataFrame([dict(panel='b',definition='Saved focal-pair order; both finite equal-weight molecular records exclude the focal pair; exact saved ten-segment steps; Pearson r on 93 unbinned conditions.'),
                    dict(panel='c',definition='Pairwise Pearson correlations of the seven and ten approved local partner-replacement patterns; diagonal omitted.'),
                    dict(panel='d',definition='Mean pairwise correlation for each molecule with at least four product-forming partners; medians 0.27/0.07 and 0.33/0.10.'),
                    dict(panel='e',definition='Pattern variance explained by amine class, compared with 4,999 molecule-level class-label shuffles.'),
                    dict(panel='FigS29',definition='Original 95-condition barcode order and missing values; retained local molecular order.')]),
            'V7_records_cells':pd.DataFrame(cells),'V7_records_steps':pd.DataFrame(steps),'V7_records_members':pd.DataFrame(members),
            'V7_record_correlations':pd.DataFrame([{k:r[k] for k in ['key','n_pairs','r','n_common']} for r in rec['records']]),
            'V7_barcode_cells':pd.DataFrame(bars),'V7_local_correlations':pd.DataFrame(payload['correlations']),
            'V7_molecule_agreement':pd.read_csv(p/'all_molecule_agreement.csv'),
            'V7_class_null_summary':pd.DataFrame(load(p/'class_null/summary.json')),
            'V7_class_null_CN':pd.read_csv(p/'class_null/cn_draws.csv'),
            'V7_class_null_amide':pd.read_csv(p/'class_null/amide_draws.csv')})
    for old,new in [('V6_class_null_summary','V7_class_null_summary'),('V6_class_null_CN','V7_class_null_CN'),('V6_class_null_amide','V7_class_null_amide')]:result[old]=result[new]
    return result

def export(template,art,source,name,number,journal,binding,rec,data):
    fonts();art.mkdir(parents=True,exist_ok=True);source.mkdir(parents=True,exist_ok=True)
    svg=art/(name+'.svg');shutil.copyfile(template,svg)
    drawing=svg2rlg(str(svg))
    renderPDF.drawToFile(drawing,str(art/(name+'.pdf')),title=name,author='',invariant=1)
    with pymupdf.open(art/(name+'.pdf')) as document:
        page=document[0]
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
        assert not page.get_images() and len(document)==1
        assert all('Arial' in s['font'] and page.rect.contains(pymupdf.Rect(s['bbox'])) for s in spans if s['text'].strip())
        assert {round(s['size'],3) for s in spans}<={5,7,8}
        page.get_pixmap(dpi=600,alpha=False).save(art/(name+'.png'))
        page.get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False).save(source/(name+'_PDF_2x.png'))
        old=load(data/'Analysis/figure_display/v7_figure3/retained_artist_readback.json')
        indices=[1,2,3,4,5,6,7,8] if name=='FigS29' else [3,7,9,10]
        axes=[dict(old['axes'][i],retained_source_axis_index=i,placement_transform_mm=([0,-40] if name=='FigS29' else [-27,53] if i==3 else [-36,53] if i==7 else [0,18])) for i in indices]
        readback=dict(schema='p2-v7-figure3-native-vector-readback-1',figure=number,journal=journal,
            width_mm=180.,height_mm=222. if name=='Figure3' else 92.,
            panel_titles={k:v for k,v in [('a','Pd-catalysed amination'),('b','Halide record follows preference'),('c','Halide retention preserves preference'),('d','Agreement across molecules'),('e','Amine class versus random')]},
            texts=[dict(text=s['text'],fontsize=s['size'],family=['Arial'],bbox_pt=s['bbox'],font=s['font']) for s in spans],
            axes=axes,labels=[dict(gid='panel_'+x,text=x,fontsize=8,weight='bold') for x in ('ab' if name=='FigS29' else 'abcde')],
            pdf_font_sizes=sorted({round(s['size'],4) for s in spans}),pdf_raster_images=0,
            pdf_vector_drawings=len(page.get_drawings()),metadata=old.get('metadata',{}),
            records=rec,source_identity=binding,native_group_identity=binding['approved_v6_native_groups_sha256'],
            panel_mapping={'a':'retained reaction','b':'frozen Figure1 record comparison','c':['axes_4','axes_8'],
                'd':'retained axes_10 +18 mm y','e':'retained axes_11 +18 mm y','FigS29':'retained old heatblocks -40 mm y'},
            scientific_values_recalculated=False)
        write_json(source/(name+'_readback.json'),readback)
    write_workbook(tables(data,rec),source/(name+'_SourceData.xlsx'))
    write_json(source/(name+'_display_identity.json'),binding)
    return readback

def cli(supplementary=False):
    ap=argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--out','--output-dir',dest='out',type=Path)
    ap.add_argument('--data-root',type=Path)
    ap.add_argument('--journal',choices=['science','natcatal'],default=os.environ.get('P2_FIGURE_JOURNAL','science'))
    args=ap.parse_args();data=(args.data_root or ROOT/'05_DATA').resolve()
    if (data/'05_DATA').is_dir():data=data/'05_DATA'
    here,binding,rec=check_assets(data)
    name='FigS29' if supplementary else 'Figure3';number=29 if supplementary else 3
    if args.out:art=source=args.out.resolve()
    elif args.journal=='natcatal':
        base=ROOT/'02_FIGURES/Journal_Formats/natcatal';art=base/('Supplementary_Figures' if supplementary else 'Main');source=base/('Supplementary_SourceData' if supplementary else 'Figure3')
    else:art=ROOT/'02_FIGURES'/('Supplementary_Figures' if supplementary else 'Main');source=ROOT/'04_SOURCE_DATA'/('Supplementary_Figures' if supplementary else 'Figure3')
    readback=export(here/(name+'_approved.svg'),art,source,name,number,args.journal,binding,rec,data)
    if not supplementary:
        supp=art if args.out else art.parent/'Supplementary_Figures'
        supp.mkdir(parents=True,exist_ok=True)
        for ext in ['svg','pdf','png']:shutil.copyfile(here/('FigS27_approved.'+ext),supp/('FigS27.'+ext))
        shutil.copyfile(here/'FigS27_readback.json',source/'FigS27_readback.json')
        with pymupdf.open(here/'FigS27_approved.pdf') as document:
            document[0].get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False).save(source/'FigS27_PDF_2x.png')
    inputs=[here/p for p in binding['frozen_files']]+[here/'BINDING.json']
    p=data/'Analysis/figure_display/v6_figure3'
    inputs += [data/relative for relative in binding['scientific_inputs']]
    files={}
    for path in inputs:files['05_DATA/'+path.relative_to(data).as_posix()]=dict(file='05_DATA/'+path.relative_to(data).as_posix(),sha256=sha(path))
    for code in ['v7_figure3.py','figs29.py' if supplementary else 'figure3.py','figure_export.py']:
        path=Path(__file__).parent/code;rel='06_CODE/Figures/'+code;files[rel]=dict(file=rel,sha256=sha(path))
    logical_art='02_FIGURES/'+('Supplementary_Figures' if supplementary else 'Main')
    logical_source='04_SOURCE_DATA/'+('Supplementary_Figures' if supplementary else 'Figure3')
    outputs={ext:dict(file=logical_art+'/'+name+'.'+ext,sha256=sha(art/(name+'.'+ext))) for ext in ['svg','pdf','png']}
    for ext in ['SourceData.xlsx','readback.json','display_identity.json','PDF_2x.png']:
        filename=name+'_'+ext;key={'SourceData.xlsx':'xlsx','readback.json':'readback'}.get(ext,ext);outputs[key]=dict(file=logical_source+'/'+filename,sha256=sha(source/filename))
    if not supplementary:
        for ext in ['svg','pdf','png']:outputs['FigS27_'+ext]=dict(file='02_FIGURES/Supplementary_Figures/FigS27.'+ext,sha256=sha(supp/('FigS27.'+ext)))
        outputs['FigS27_readback.json']=dict(file='04_SOURCE_DATA/Figure3/FigS27_readback.json',sha256=sha(source/'FigS27_readback.json'))
        outputs['FigS27_PDF_2x.png']=dict(file='04_SOURCE_DATA/Figure3/FigS27_PDF_2x.png',sha256=sha(source/'FigS27_PDF_2x.png'))
    retained=load(here/'retained_scientific_metadata.json')
    provenance=dict(figure='Fig. S29' if supplementary else 'Figure 3',generated_by='06_CODE/Figures/'+('figs29.py' if supplementary else 'figure3.py'),
                    inputs={'files':files},outputs=outputs,panels=retained['source_panels'],
                    panel_metadata_scope=retained['scope'],panel_mapping=readback['panel_mapping'])
    write_json(source/(name+'_SourceData_provenance.json'),provenance)
    print(json.dumps(dict(figure=name,status='PASS',journal=args.journal,output=str(art)),indent=2))

if __name__=='__main__':cli()
