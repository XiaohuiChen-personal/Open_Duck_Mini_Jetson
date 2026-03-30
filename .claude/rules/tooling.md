# Tooling

## Exporting draw.io Diagrams to PNG

The wiring diagram at `docs/jetson-mod/jetson_wiring_diagram.drawio` has a corresponding PNG render at `docs/jetson-mod/jetson_wiring_diagram.png`. When the `.drawio` file is modified, the PNG must be re-exported.

### Method: Headless Export via Puppeteer + draw.io Viewer

This machine has no draw.io desktop app. Use the npm `@mattiash/drawio-export` package or the following puppeteer-based approach:

```bash
# Requires: node, npm, chromium (all available on DGX Spark)
# One-time setup (in /tmp):
cd /tmp && npm install puppeteer

# Export script:
cat > /tmp/export_drawio.mjs << 'JSEOF'
import puppeteer from 'puppeteer';
import fs from 'fs';

const drawioFile = process.argv[2];
const outputFile = process.argv[3];
const xmlContent = fs.readFileSync(drawioFile, 'utf8');
const match = xmlContent.match(/<mxGraphModel[\s\S]*?<\/mxGraphModel>/);
if (!match) { console.error('No mxGraphModel found'); process.exit(1); }
const b64 = Buffer.from(match[0]).toString('base64');

const html = `<!DOCTYPE html>
<html><head><style>body{margin:0;padding:10px;background:white;}</style></head><body>
<div id="graph" class="mxgraph"></div>
<script>
var div=document.getElementById('graph');
div.setAttribute('data-mxgraph',JSON.stringify({highlight:'#0000ff',nav:false,resize:true,toolbar:'',edit:'_blank',xml:atob("${b64}")}));
</script>
<script src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>
</body></html>`;

const browser = await puppeteer.launch({
  executablePath: '/snap/bin/chromium',
  headless: 'new',
  args: ['--no-sandbox','--disable-setuid-sandbox','--disable-gpu']
});
const page = await browser.newPage();
await page.setViewport({width:2200,height:1800});
await page.setContent(html,{waitUntil:'networkidle0',timeout:30000});
await new Promise(r=>setTimeout(r,6000));
await page.screenshot({path:outputFile,type:'png',fullPage:true});
console.log('Exported '+fs.statSync(outputFile).size+' bytes');
await browser.close();
JSEOF

# Run:
cd /tmp && node /tmp/export_drawio.mjs \
  /path/to/diagram.drawio \
  /path/to/output.png
```

### When to Re-export

Re-export the PNG whenever `jetson_wiring_diagram.drawio` is modified (e.g., voltage changes, component additions). The PNG is used in documentation and should always match the drawio source.
