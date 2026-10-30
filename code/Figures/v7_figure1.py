"""Native layered SVG and embedded-Arial PDF for the v7 map with its v8 display amendment.

Rebuilds use frozen display bindings and ChemDraw vectors only. No outcomes are
read or recomputed. The one-time identity selection is recorded in the rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
import numpy as np
import pymupdf
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen.canvas import Canvas
from v6_figure1 import fonts
from structure_art import load as load_art

ROOT = Path(__file__).resolve().parents[2]
DISPLAY = "05_DATA/Analysis/figure_display/v7_figure1"
RECORD_SHA = "515e2813b4d18c6b93177480f9192cdbeb08c1929ec2bf2f9511433916e9697c"
AMENDMENT = DISPLAY + "/Figure1_DISPLAY_AMENDMENT_v8.json"
AMENDMENT_SHA = "cb361f14841d11fc567b439414ba1822da548a40edbfdd48e6cd353427abb382"
NS, INK = "http://www.w3.org/2000/svg", "http://www.inkscape.org/namespaces/inkscape"
PT = 72 / 25.4
WIDTH, HEIGHT = 180., 130.
BLUE, ORANGE, DARK, LIGHT = "#0072B2", "#D55E00", "#4D4D4D", "#B3B3B3"
BLACK, GREY = "#181818", "#777777"
ET.register_namespace("", NS)
ET.register_namespace("inkscape", INK)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Inputs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.reads = {}

    def path(self, relative):
        candidates = [self.root / relative]
        aliases = [("05_DATA/Analysis/figure_display/", "figure_inputs/"),
                   ("02_FIGURES/ChemDraw/", "figures/structures/"),
                   ("04_SOURCE_DATA/", "source_data/"),("06_CODE/", "code/")]
        roots = [self.root]
        if self.root.parent.name == "_rebuild":
            roots.append(self.root.parent.parent)
        for base in roots:
            for old, new in aliases:
                if relative.startswith(old):
                    candidates.append(base / (new + relative[len(old):]))
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Released Figure 1 input missing: {relative}")
        self.reads[path] = relative
        return path

    def json(self, relative):
        return json.loads(self.path(relative).read_text(encoding="utf-8"))


class Scene:
    def __init__(self, inputs, records, journal, amendment):
        self.inputs, self.records, self.journal = inputs, records, journal
        self.svg = ET.Element(f"{{{NS}}}svg", width="180mm", height="130mm", viewBox="0 0 180 130")
        self.el(self.svg, "title").text = amendment["title"]
        self.el(self.svg, "desc").text = (
            "Field-level molecular experience and the untested owner question lead to reciprocal partner replacements, "
            "molecular versus pair-specific decomposition, and the discoveries of median 80% molecular ownership, "
            "reaction-dependent roles, halide ownership and approximately threefold condition-profile prediction. Small real recurring molecules illustrate "
            "identity connections only. Schematic well counts encode no measurements.")
        self.texts, self.primitives, self.structures = [], [], []

    @staticmethod
    def el(parent, tag, **attrs):
        return ET.SubElement(parent, f"{{{NS}}}{tag}", {k.replace("_", "-"): str(v) for k, v in attrs.items()})

    def layer(self, name):
        group = self.el(self.svg, "g", id=name)
        group.set(f"{{{INK}}}groupmode", "layer")
        group.set(f"{{{INK}}}label", name)
        return group

    def text(self, group, x, y, value, size=7, color=BLACK, anchor="start", bold=False, id=None, role="annotation"):
        attrs = dict(x=x, y=y, font_family="Arial", font_size=size/PT, fill=color,
                     text_anchor=anchor, font_weight="bold" if bold else "normal", font_style="normal", data_role=role)
        if id:
            attrs["id"] = id
        node = self.el(group, "text", **attrs)
        node.text = value
        font = "Arial-Bold" if bold else "Arial"
        width = pdfmetrics.stringWidth(value, font, size)/PT
        left = x-width*{"start":0, "middle":.5, "end":1}[anchor]
        ascent, descent = pdfmetrics.getAscentDescent(font, size)
        self.texts.append(dict(id=id, text=value, fontsize=size, family=["Arial"], weight="bold" if bold else "normal",
                               color=color, role=role, bbox_mm=[left,y-ascent/PT,left+width,y-descent/PT]))
        return node

    def cited(self, group, x, y, value, key):
        self.text(group,x,y,value,size=5,color=GREY,id=key)
        width = pdfmetrics.stringWidth(value,"Arial",5)/PT
        self.text(group,x+width+.1,y-.7,self.records["citations"][key],size=5,color=GREY,id=key+"_citation",role="citation")

    def line(self, group, points, color=GREY, width=.18, arrow=False, id=None, role="route"):
        attrs = dict(points=" ".join(f"{x:.8f},{y:.8f}" for x,y in points),fill="none",stroke=color,stroke_width=width,data_role=role)
        if id:
            attrs["id"] = id
        self.el(group,"polyline",**attrs)
        self.primitives.append(dict(id=id,role=role,points=points,width_mm=width))
        if arrow:
            end,prior=np.asarray(points[-1],float),np.asarray(points[-2],float)
            direction=(end-prior)/np.linalg.norm(end-prior);normal=np.asarray([-direction[1],direction[0]])
            triangle=[end,end-1.2*direction+.42*normal,end-1.2*direction-.42*normal]
            self.el(group,"polygon",points=" ".join(f"{x:.8f},{y:.8f}"for x,y in triangle),fill=color)

    def rect(self, group, x,y,w,h,fill="none",stroke="none",id=None,role="shape",**attrs):
        values=dict(x=x,y=y,width=w,height=h,fill=fill,stroke=stroke,stroke_width=.18,data_role=role,**attrs)
        if id:
            values["id"]=id
        self.el(group,"rect",**values)
        self.primitives.append(dict(id=id,role=role,bbox_mm=[x,y,x+w,y+h]))

    def well(self,group,x,y,r=.9):
        self.el(group,"circle",cx=x,cy=y,r=r,fill="none",stroke=LIGHT,stroke_width=.18)
        self.primitives.append(dict(role="schematic_well",center_mm=[x,y],radius_mm=r))

    def structure(self, group, key, cx,cy,id):
        entry=next(x for x in self.records["vectors"]["items"]if x["name"]==key)
        rel=f"02_FIGURES/ChemDraw/v7/f1/{key}.svg"
        path=self.inputs.path(rel);cdxml=self.inputs.path(rel.replace('.svg','.cdxml'))
        assert sha(path)==entry["svg_sha256"] and sha(cdxml)==entry["cdxml_sha256"]
        art=load_art(path);k=7/art["label_px"]/PT
        x,y=cx-art["width"]*k/2,cy-art["height"]*k/2
        group=self.el(group,"g",id=id,data_master=rel,data_master_sha256=sha(path),data_smiles=entry["smiles"],data_atom_pt=7,data_uniform_scale=k)
        for poly in art["polygons"]:
            coords=poly*k+(x,y)
            self.el(group,"polygon",points=" ".join(f"{a:.9f},{b:.9f}"for a,b in coords),fill="black",stroke="black",stroke_width=.08/PT)
        for i,label in enumerate(art["labels"]):
            self.text(group,x+label["x"]*k,y+label["y"]*k,label["text"],size=7*label["scale"],color="black",id=f"{id}_atom_{i}",role="atom")
        self.structures.append(dict(id=id,key=key,smiles=entry["smiles"],master=rel,sha256=sha(path),cdxml_sha256=sha(cdxml),atom_pt=7,
                                    uniform_scale=k,bbox_mm=[x,y,x+art["width"]*k,y+art["height"]*k]))


def make_scene(inputs,records,journal,amendment):
    fonts();s=Scene(inputs,records,journal,amendment)
    headlines={entry['id']:entry['text']for entry in amendment['right_column_headlines']}
    need=s.layer("01_necessity_and_challenge");route=s.layer("02_reciprocal_replacement_route")
    discoveries=s.layer("03_discoveries");reading=s.layer("04_reading_path")
    for group,x,letter,title in [(need,4,"a","Whose knowledge transfers?"),(route,58,"b","Replace either partner"),(discoveries,121,"c","One molecule dominates")]:
        s.text(group,x,6,letter,size=8,bold=True,id="panel_"+letter,role="panel")
        s.text(group,x+4.4,6,title,id="panel_"+letter+"_title")
    s.line(reading,[(52.5,65),(56.3,65)],color=DARK,arrow=True)
    s.line(reading,[(115.1,65),(118.7,65)],color=DARK,arrow=True)
    s.text(need,4,14,"Chemists reuse molecular experience")
    s.text(need,4,25,"Ar–X + H–Nu",id="generic_coupling")
    s.line(need,[(26,24),(40,24)],color=BLACK,arrow=True)
    s.text(need,33,20,"[M] / L",size=5,anchor="middle")
    s.text(need,33,30,"base / solvent",size=5,anchor="middle")
    s.text(need,43,25,"Ar–Nu")
    s.cited(need,4,19,"Amine-based ligand guides","cite_ligand_guides")
    s.line(need,[(27.2,18.4),(29.4,19)],color=LIGHT,id="guide_leader")
    s.cited(need,4,31,"General conditions","cite_general_conditions")
    s.line(need,[(22.7,30.4),(25.5,29.4)],color=LIGHT,id="general_leader")
    s.text(need,4,40,"Molecules meet new partners")
    s.structure(need,"recurring_halide",14,67,"recurring_aryl_bromide")
    for key,cx,cy in [("recurring_amine1",41,51),("recurring_amine2",41,70),("recurring_amine3",41,88)]:
        s.structure(need,key,cx,cy,key)
    for endpoint in [(34.5,51),(33,70),(31.5,88)]:
        s.line(need,[(25,67),endpoint],color=LIGHT,id="identity_link_"+str(endpoint[1]),role="identity_connection")
    s.cited(need,4,56,"Informer libraries","cite_informer")
    s.line(need,[(12,57.5),(12,59.9)],color=LIGHT,id="informer_leader")
    s.cited(need,4,86,"Robustness screens","cite_robustness")
    s.line(need,[(23,85.4),(30.3,87.5)],color=LIGHT,id="robustness_leader")
    s.text(need,4,103,"63% recur with partners",id="recurrence_63pct")
    s.text(need,4,108,"Molecules with >1 partner",size=5,color=GREY)
    s.line(need,[(4,112),(50,112)],color=LIGHT)
    s.text(need,4,118,"Which molecule carries the conditions?",id="knowledge_gap")
    s.text(need,4,124,"Never tested",color=DARK)

    s.text(route,58,17,"Every pair, same conditions")
    s.text(route,77,24,"Molecule B",size=5,color=ORANGE,anchor="middle")
    s.text(route,58.8,37,"A",color=BLUE)
    for i in range(4):
        for j in range(4):s.well(route,65+6.1*j,28+6.1*i)
    s.rect(route,63.5,32.5,21.3,3.2,stroke=BLUE,role="fixed_A")
    s.rect(route,69.6,26.5,3.2,21.3,stroke=ORANGE,role="fixed_B")
    s.line(route,[(85.8,34),(99.1,34)],arrow=True)
    s.text(route,106.3,26.5,"Each pair",size=5,anchor="middle")
    for i in range(3):
        for j in range(3):s.well(route,101.5+3.2*j,30.5+3.2*i,r=.67)
    s.text(route,105.5,43.5,"conditions",size=5,anchor="middle")
    s.text(route,86,53.5,"177,039 reactions",anchor="middle",id="reaction_count")
    s.text(route,86,59,"15 screens · 14 transformations",anchor="middle",id="scope")
    s.text(route,71,68,"Fix A; replace B",size=5,color=BLUE,anchor="middle")
    s.text(route,101,68,"Fix B; replace A",size=5,color=ORANGE,anchor="middle")
    for cx,color,fixed in [(71,BLUE,"row"),(101,ORANGE,"column")]:
        for i in range(3):
            for j in range(3):s.well(route,cx-4.3+4.3*j,74+4.3*i,r=.8)
        if fixed=="row":
            s.rect(route,cx-5.7,76.9,11.4,2.8,stroke=color,role="fixed_A")
            s.line(route,[(cx-5.5,86),(cx+5.5,86)],color=color,arrow=True)
        else:
            s.rect(route,cx-1.4,72.6,2.8,11.4,stroke=color,role="fixed_B")
            s.line(route,[(cx+8.4,73),(cx+8.4,83)],color=color,arrow=True)
    s.text(route,86,94,"Molecular vs pair-specific",anchor="middle")
    for x,label,color in [(67.5,"A",BLUE),(81,"B",ORANGE),(103.5,"Pair-specific",DARK)]:
        s.line(route,[(86,96.2),(x,100)],color=LIGHT)
        s.text(route,x,104,label,size=5,color=color,anchor="middle")
    s.line(route,[(63,105),(63,106),(85.5,106),(85.5,105)])
    s.text(route,74.3,110.5,"Molecular share",size=5,anchor="middle")
    s.line(route,[(86,112.5),(86,115)],arrow=True)
    s.text(route,86,120,"844 outside-screen pairs",anchor="middle",id="outside_pairs")
    s.text(route,86,126,"2 external public screens",size=5,anchor="middle",id="external_screens")

    s.text(discoveries,121,18,headlines['median_ownership'],id="median_ownership")
    s.text(discoveries,121,23,"Owner / molecular component",size=5,color=GREY,id="median_denominator")
    s.text(discoveries,121,34,headlines['role_switch_title'],id="role_switch_title")
    for cx,label,role,color in [(135.2,"Amide coupling","Owner",ORANGE),(163.6,"C–N coupling","Partner",DARK)]:
        s.structure(discoveries,"role_switch",cx,49.2,"role_switch_"+role.lower())
        s.text(discoveries,cx,62,label,size=5,anchor="middle")
        s.text(discoveries,cx,66,role,size=5,anchor="middle",color=color)
    s.text(discoveries,149.4,70,"4-methylaniline",size=5,anchor="middle",color=GREY)
    s.text(discoveries,121,82,headlines['halide_discovery'],id="halide_discovery")
    blocks=records["owner_blocks"]
    assert len(blocks)==15 and [b["colour"]for b in blocks]==[BLUE]*10+[ORANGE,DARK,DARK,DARK,ORANGE]
    for i,block in enumerate(blocks):
        s.rect(discoveries,122+i*3.5,88,2.75,3.1,fill=block["colour"],id=f"owner_role_{i+1:02d}",role="owner_identity",data_screen=block["screen"],data_owner=block["owner"])
    for x,label,color in [(122,"halide",BLUE),(140,"amine / N–H",ORANGE),(164,"other",DARK)]:
        s.text(discoveries,x,98,label,size=5,color=color)
    s.text(discoveries,121,104,"Order: Fig. 2a",size=5,color=GREY,id="owner_source")
    s.text(discoveries,121,110,headlines['prediction_discovery'],id="prediction_discovery")
    s.text(discoveries,121,114.5,"Preference explained",size=5,color=GREY)
    transfer={row["side"]:row for row in records["transfer"]["rows"]}
    for side,y,color in [("owner",118.8,DARK),("partner",124,LIGHT)]:
        s.text(discoveries,121,y+.6,side.capitalize(),size=5,color=DARK)
        s.rect(discoveries,143,y-.45,float(transfer[side]["value"])*1.02,.9,fill=color,id="transfer_"+side,role="prediction_bar",data_value_percent=transfer[side]["value"])
    s.text(discoveries,176,129,"Set A · Fig. 5a",size=5,color=GREY,anchor="end",id="transfer_source")
    return s


def render_pdf(scene,path):
    canvas=Canvas(str(path),pagesize=(WIDTH*PT,HEIGHT*PT),invariant=1,pageCompression=1,initialFontName="Arial",initialFontSize=7)
    canvas.setTitle("Figure 1");canvas.setAuthor("");canvas.setCreator("06_CODE/Figures/figure1.py")
    for node in scene.svg.iter():
        tag=node.tag.split('}')[-1];a=node.attrib
        if tag not in ('text','polyline','polygon','rect','circle'):continue
        fill,stroke=a.get('fill','none'),a.get('stroke','none')
        if fill!='none':canvas.setFillColor(fill)
        if stroke!='none':canvas.setStrokeColor(stroke);canvas.setLineWidth(float(a.get('stroke-width',.18))*PT)
        if tag=='text':
            canvas.setFont('Arial-Bold'if a.get('font-weight')=='bold'else'Arial',float(a['font-size'])*PT)
            {'start':canvas.drawString,'middle':canvas.drawCentredString,'end':canvas.drawRightString}[a.get('text-anchor','start')](float(a['x'])*PT,(HEIGHT-float(a['y']))*PT,node.text or '')
        elif tag=='rect':
            x,y,w,h=[float(a[k])for k in ('x','y','width','height')]
            canvas.rect(x*PT,(HEIGHT-y-h)*PT,w*PT,h*PT,fill=int(fill!='none'),stroke=int(stroke!='none'))
        elif tag=='circle':
            canvas.circle(float(a['cx'])*PT,(HEIGHT-float(a['cy']))*PT,float(a['r'])*PT,fill=int(fill!='none'),stroke=int(stroke!='none'))
        else:
            points=[[float(v)for v in p.split(',')]for p in a['points'].split()]
            p=canvas.beginPath();p.moveTo(points[0][0]*PT,(HEIGHT-points[0][1])*PT)
            for x,y in points[1:]:p.lineTo(x*PT,(HEIGHT-y)*PT)
            if tag=='polygon':p.close()
            canvas.drawPath(p,fill=int(fill!='none'),stroke=int(stroke!='none'))
    canvas.showPage();canvas.save()


def generate(out=None,journal="science",data_root=None):
    inputs=Inputs(data_root or ROOT)
    record_path=inputs.path(DISPLAY+"/records.json");assert sha(record_path)==RECORD_SHA
    records=json.loads(record_path.read_text(encoding='utf-8'))
    assert sha(inputs.path(records['rule']['path']))==records['rule']['sha256']
    assert sha(inputs.path(records['selection']['path']))==records['selection']['sha256']
    assert sha(inputs.path(records['vectors']['manifest']))==records['vectors']['manifest_sha256']
    amendment=inputs.json(AMENDMENT);assert sha(inputs.path(AMENDMENT))==AMENDMENT_SHA
    assert amendment['source_records_sha256']==RECORD_SHA and amendment['original_rule_sha256']==records['rule']['sha256']
    scene=make_scene(inputs,records,journal,amendment)
    artdir=Path(out)if out else ROOT/('02_FIGURES/Main'if journal=='science'else'02_FIGURES/Journal_Formats/natcatal/Main')
    datadir=Path(out)if out else ROOT/('04_SOURCE_DATA/Figure1'if journal=='science'else'02_FIGURES/Journal_Formats/natcatal/Figure1')
    artdir.mkdir(parents=True,exist_ok=True);datadir.mkdir(parents=True,exist_ok=True)
    metadata=dict(figure=1,version='v7_formal',journal_invariant=True,width_mm=WIDTH,height_mm=HEIGHT,panels=['a','b','c'],
                  display_revision=amendment['revision'],display_title=amendment['title'],display_amendment_sha256=AMENDMENT_SHA,
                  panel_titles={x['id']:x['text']for x in scene.texts if (x['id']or'').endswith('_title')},
                  scientific_values_recalculated=False,records_input=DISPLAY+'/records.json',source_records_sha256=RECORD_SHA,
                  rule_sha256=records['rule']['sha256'],texts=scene.texts,text_styles=scene.texts,structures=scene.structures,primitives=scene.primitives,
                  owner_blocks=records['owner_blocks'],transfer=records['transfer'],citation_numbers=records['citations'],
                  primary_discovery_numbers=amendment['primary_discovery_numbers'],global_atom_pt=7,master_bond_pt=14.4,
                  instruction_ruling='Latest explicit formal request for real recurrence structures in panel a overrides older only-concrete-molecule-in-right sentence.',
                  schematic_counts_are_data=False)
    for text in scene.texts:
        if text['role'] not in ('atom','citation'):assert len(text['text'].split())<=5,text['text']
        x1,y1,x2,y2=text['bbox_mm'];assert 0<=x1<x2<=WIDTH and 0<=y1<y2<=HEIGHT,text
    scene.el(scene.svg,'metadata',id='v7_display_binding').text=json.dumps(metadata,ensure_ascii=False,sort_keys=True)
    ET.ElementTree(scene.svg).write(artdir/'Figure1.svg',encoding='utf-8',xml_declaration=True)
    render_pdf(scene,artdir/'Figure1.pdf')
    with pymupdf.open(artdir/'Figure1.pdf')as doc:
        doc[0].get_pixmap(dpi=600,alpha=False).save(artdir/'Figure1.png')
        doc[0].get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False).save(datadir/'Figure1_v7_PDF_2x.png')
        metadata['pdf_fonts']=[list(x)for x in doc[0].get_fonts(full=True)]
        metadata['pdf_raster_images']=len(doc[0].get_images(full=True))
        assert metadata['pdf_raster_images']==0
    (datadir/'Figure1_v7_readback.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    binding=dict(records=records,records_sha256=RECORD_SHA,layout=dict(width_mm=WIDTH,height_mm=HEIGHT,columns_mm=[[4,50],[58,114],[121,176]]),
                 display_amendment=amendment,display_amendment_sha256=AMENDMENT_SHA,
                 occurrence=dict(owner_blocks=[dict(svg_id=f'owner_role_{i+1:02d}',**b)for i,b in enumerate(records['owner_blocks'])],
                 Set_A=dict(owner_svg_id='transfer_owner',partner_svg_id='transfer_partner',values_caption_only=True)),
                 instructions=metadata['instruction_ruling'])
    (datadir/'Figure1_display_bindings.json').write_text(json.dumps(binding,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    # Retain legacy scientific tables/payload byte-for-byte; new map gets explicit separate bindings.
    legacy=[]
    for entry in records['legacy']:
        name=entry['file'];source=inputs.path(entry['path']);target=datadir/name
        assert sha(source)==entry['sha256']
        if source.resolve()!=target.resolve():shutil.copyfile(source,target)
        legacy.append(dict(file=name,sha256=sha(source),status='unchanged legacy scientific evidence; v7 map uses explicit display bindings'))
    shutil.copyfile(inputs.path(records['rule']['path']),datadir/'Figure1_DISPLAY_RULE.md')
    shutil.copyfile(inputs.path(AMENDMENT),datadir/'Figure1_DISPLAY_AMENDMENT_v8.json')
    for relative in ['06_CODE/Figures/figure1.py','06_CODE/Figures/v7_figure1.py','06_CODE/Figures/v6_figure1.py','06_CODE/Figures/structure_art.py']:
        inputs.path(relative)
    destinations={
      'svg':('02_FIGURES/Main/Figure1.svg',artdir/'Figure1.svg'),
      'pdf':('02_FIGURES/Main/Figure1.pdf',artdir/'Figure1.pdf'),
      'png':('02_FIGURES/Main/Figure1.png',artdir/'Figure1.png')}
    for key,name in [('xlsx','Figure1_SourceData.xlsx'),('readback','Figure1_readback.json'),('payload','figure1_payload.json'),
                     ('v7_readback','Figure1_v7_readback.json'),('display_bindings','Figure1_display_bindings.json'),
                     ('display_rule','Figure1_DISPLAY_RULE.md'),('display_amendment','Figure1_DISPLAY_AMENDMENT_v8.json'),('pdf_preview','Figure1_v7_PDF_2x.png')]:
        destinations[key]=('04_SOURCE_DATA/Figure1/'+name,datadir/name)
    provenance=dict(figure='Figure 1',generated_by='06_CODE/Figures/figure1.py',version='v7_formal',journal_invariant=True,
      display_revision=amendment['revision'],display_amendment_sha256=AMENDMENT_SHA,
      inputs={'files':{relative:dict(file=relative,sha256=sha(path))for path,relative in sorted(inputs.reads.items(),key=lambda item:item[1])}},
      outputs={key:dict(file=relative,sha256=sha(actual))for key,(relative,actual)in destinations.items()},
      panels={'a':'necessity and challenge','b':'reciprocal partner replacements','c':'discoveries'},
      legacy_preserved=legacy,scientific_values_recalculated=False)
    (datadir/'Figure1_SourceData_provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(figure=1,status='PASS',journal=journal,output=str(artdir),data=str(datadir),panels=3,owner_identity_blocks=15,prediction_bars=2,pdf_raster_images=0),indent=2))
    return metadata


def cli():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--out','--output-dir',dest='out',type=Path)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--journal',choices=('science','natcatal'),default=os.environ.get('P2_FIGURE_JOURNAL','science'))
    args=parser.parse_args();generate(args.out,args.journal,args.data_root)


if __name__=='__main__':cli()
