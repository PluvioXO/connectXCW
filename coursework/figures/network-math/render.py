"""Compile the figure's LaTeX maths to self-contained, outlined SVG assets."""
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

SOURCE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).with_name('network-math.tex')
ROOT = SOURCE.parent
TEX_BIN = Path('/Library/TeX/texbin')
SVG = 'http://www.w3.org/2000/svg'
XLINK = 'http://www.w3.org/1999/xlink'
ET.register_namespace('', SVG)


def expand(node, definitions, seen=()):
    if node.tag == f'{{{SVG}}}use':
        ref = node.attrib[f'{{{XLINK}}}href']
        assert ref.startswith('#') and ref not in seen, ref
        attrs = {k: v for k, v in node.attrib.items()
                 if k not in {'id', 'x', 'y', f'{{{XLINK}}}href'}}
        attrs['transform'] = (attrs.get('transform', '') +
                              f" translate({node.get('x', '0')},{node.get('y', '0')})").strip()
        group = ET.Element(f'{{{SVG}}}g', attrs)
        group.append(expand(definitions[ref[1:]], definitions, (*seen, ref)))
        return group
    result = copy.copy(node)
    result.attrib = dict(node.attrib)
    result.attrib.pop('id', None)
    result[:] = [expand(child, definitions, seen) for child in node]
    return result


manifest = json.loads((ROOT / 'manifest.json').read_text())
with tempfile.TemporaryDirectory(prefix='connectx-math-') as temp:
    work = Path(temp)
    commands = [
        [str(TEX_BIN / 'xelatex'), '-no-pdf', '-interaction=nonstopmode',
         '-halt-on-error', f'-output-directory={work}', str(SOURCE)],
        [str(TEX_BIN / 'dvisvgm'), '--no-fonts', '--exact-bbox', '--page=1-',
         f'--output={work}/eq-%p.svg', str(work / SOURCE.with_suffix('.xdv').name)],
    ]
    for command in commands:
        run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if run.returncode:
            raise RuntimeError(run.stdout)
    pages = sorted(work.glob('eq-*.svg'))
    assert len(pages) == len(manifest['items']), 'LaTeX page/manifest mismatch'
    for item, path in zip(manifest['items'], pages):
        svg = ET.parse(path).getroot()
        definitions = {n.attrib['id']: n for n in svg.findall(f'{{{SVG}}}defs/*')}
        for child in list(svg):
            if child.tag == f'{{{SVG}}}defs':
                svg.remove(child)
        svg = expand(svg, definitions)
        _, _, width, height = map(float, svg.attrib['viewBox'].split())
        scale = min(1., item['maxWidth'] / width, item['slotHeight'] / height)
        assert scale >= .9, (item['key'], 'Math no longer fits', scale)
        item.update(width=round(width * scale, 4), height=round(height * scale, 4),
                    scale=round(scale, 6), svgFile=item['key'] + '.svg')
        svg.set('width', str(item['width']))
        svg.set('height', str(item['height']))
        svg.set('role', 'img')
        svg.set('aria-label', item['latex'])
        assert not any(n.tag in {f'{{{SVG}}}{tag}' for tag in ['text', 'image', 'use']}
                       for n in svg.iter()), item['key']
        (ROOT / item['svgFile']).write_text(ET.tostring(svg, encoding='unicode') + '\n')

(ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f"Rendered {len(manifest['items'])} outlined SVG assets; no external fonts or images.")
