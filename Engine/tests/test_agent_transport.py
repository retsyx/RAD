"""MCP stdio transport and image-path hardening of the agent tool server."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from agent_server import read_regular, transport_result


class MCPTransportTests(unittest.TestCase):
    def prepare(self, folder):
        bridge = folder / '.rad-tools'
        (bridge / 'requests').mkdir(parents=True)
        (bridge / 'responses').mkdir()
        return bridge

    def test_stdio_protocol_discovers_tools_and_returns_real_image_content(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            bridge = self.prepare(folder)
            # Header is enough for transport: native render tests validate actual PNGs.
            image = folder / 'render.png'
            image.write_bytes(b'\x89PNG\r\n\x1a\nexample')
            process = subprocess.Popen([sys.executable, str(ENGINE / 'agent_server.py'), '--workspace', str(folder)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            def call(message):
                process.stdin.write(json.dumps(message) + '\n'); process.stdin.flush()
                return json.loads(process.stdout.readline())
            try:
                initialized = call({'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2024-11-05'}})
                self.assertIn('no CAD bootstrap', initialized['result']['instructions'])
                tools = call({'jsonrpc':'2.0','id':2,'method':'tools/list'})['result']['tools']
                names = {tool['name'] for tool in tools}
                catalog = json.loads((ENGINE / 'agent-tools.json').read_text(encoding='utf-8'))
                self.assertEqual(tools, [{key: entry[key] for key in ('name', 'description', 'inputSchema', 'annotations')}
                                         for entry in catalog], 'Agents see the catalog without its host properties.')
                self.assertTrue({'rad_build','rad_inspect','rad_measure','rad_render','rad_screenshot','rad_validate','rad_runtime'} <= names)
                self.assertIn('rad_export_3mf', names)
                self.assertTrue({'rad_export_stl', 'rad_export_step'}.isdisjoint(names), 'SDF designs export 3MF')
                self.assertIn('rad_import', names, 'STL and OBJ meshes are imported as SDF solids')
                self.assertFalse({name for name in names if name.startswith('rad_mesh_')}, 'Mesh reconstruction tools are gone')
                geometry_tools = {'rad_runtime', 'rad_build', 'rad_inspect', 'rad_measure', 'rad_validate',
                                  'rad_check_motion', 'rad_export_3mf'}
                for tool in tools:
                    claims = ('CadQuery', 'OpenCascade', 'STEP', 'BREP') + (('exact',) if tool['name'] in geometry_tools else ())
                    for claim in claims:
                        self.assertNotIn(claim, tool['description'], tool['name'])
                for claim in ('CadQuery', 'OpenCascade', 'STEP', 'BREP'):
                    self.assertNotIn(claim, initialized['result']['instructions'])
                def native():
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        requests = list((bridge/'requests').glob('*.json'))
                        if requests:
                            request = json.loads(requests[0].read_text())
                            self.assertEqual(request['name'], 'rad_render')
                            (bridge/'responses'/(request['id']+'.json')).write_text(json.dumps({'content':[{'type':'text','text':'{"artifactID":"test"}'},{'type':'image_path','path':str(image),'mimeType':'image/png'}]}))
                            return
                        time.sleep(0.01)
                thread = threading.Thread(target=native); thread.start()
                result = call({'jsonrpc':'2.0','id':3,'method':'tools/call','params':{'name':'rad_render','arguments':{'artifactID':'test'}}})['result']
                thread.join()
                self.assertEqual(result['content'][1]['type'], 'image')
                self.assertEqual(base64.b64decode(result['content'][1]['data']), image.read_bytes())
                self.assertFalse(list((bridge/'responses').glob('*.json')))
                (bridge/'closed').write_text('closed')
                failed = call({'jsonrpc':'2.0','id':4,'method':'tools/call','params':{'name':'rad_runtime','arguments':{}}})['result']
                self.assertTrue(failed['isError'])
                self.assertIn('stopped', failed['content'][0]['text'])
            finally:
                process.stdin.close()
                process.wait(timeout=3)
                self.assertEqual(process.returncode, 0, process.stderr.read())
                process.stdout.close(); process.stderr.close()

    def test_image_adapter_rejects_external_paths_symlinks_and_hardlinks(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            folder = Path(directory)
            external = Path(outside)/'image.png'; external.write_bytes(b'\x89PNG\r\n\x1a\nimage')
            for path in [external, folder/'symlink.png', folder/'hardlink.png']:
                if path.name == 'symlink.png': path.symlink_to(external)
                if path.name == 'hardlink.png': os.link(external, path)
                with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                    transport_result(folder, {'content':[{'type':'image_path','path':str(path)}]})
            link = folder/'linked-dir'; link.symlink_to(Path(outside), target_is_directory=True)
            with self.assertRaises(OSError):
                read_regular(folder, link/'image.png', 1024)


if __name__ == '__main__':
    unittest.main()
