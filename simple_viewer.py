"""
Simple HTML viewer for ErgonomicXAI results
"""
import json
import os
from pathlib import Path
import webbrowser
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

class ResultsHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Generate HTML content
            html_content = self.generate_results_html()
            self.wfile.write(html_content.encode())
        else:
            super().do_GET()
    
    def generate_results_html(self):
        results_dir = Path("validation_results/run_subset")
        json_files = list(results_dir.glob("*.json"))
        
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>ErgonomicXAI Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
        .result-card { border: 1px solid #ddd; margin: 15px 0; padding: 20px; border-radius: 8px; background: #fafafa; }
        .risk-high { border-left: 5px solid #e74c3c; }
        .risk-medium { border-left: 5px solid #f39c12; }
        .risk-low { border-left: 5px solid #27ae60; }
        .score { font-size: 24px; font-weight: bold; margin: 10px 0; }
        .parts { display: flex; gap: 20px; margin: 10px 0; }
        .part { background: #ecf0f1; padding: 10px; border-radius: 5px; text-align: center; }
        .advice { background: #e8f4fd; padding: 15px; border-radius: 5px; margin-top: 10px; }
        .chart { width: 100%; height: 200px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🦾 ErgonomicXAI Analysis Results</h1>
            <p>Comprehensive ergonomic risk assessment for manufacturing images</p>
        </div>
"""
        
        # Process each JSON file
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Determine risk level
                reba_score = data.get('reba', 0)
                if reba_score > 6:
                    risk_class = "risk-high"
                    risk_level = "High Risk"
                elif reba_score >= 3:
                    risk_class = "risk-medium"
                    risk_level = "Medium Risk"
                else:
                    risk_class = "risk-low"
                    risk_level = "Low Risk"
                
                # Get parts breakdown
                parts = data.get('parts', {})
                advice = data.get('advice', 'No advice available')
                
                html += f"""
        <div class="result-card {risk_class}">
            <h3>📸 {data.get('image', 'Unknown Image')}</h3>
            <div class="score">REBA Score: {reba_score:.1f} ({risk_level})</div>
            <div class="parts">
                <div class="part">
                    <strong>Trunk</strong><br>
                    {parts.get('trunk', 0):.1%}
                </div>
                <div class="part">
                    <strong>Arms</strong><br>
                    {parts.get('arms', 0):.1%}
                </div>
                <div class="part">
                    <strong>Legs</strong><br>
                    {parts.get('legs', 0):.1%}
                </div>
            </div>
            <div class="advice">
                <strong>💡 Actionable Advice:</strong><br>
                {advice.replace('**', '').replace('\\n', '<br>')}
            </div>
        </div>
"""
            except Exception as e:
                html += f"""
        <div class="result-card">
            <h3>Error processing {json_file.name}</h3>
            <p>Error: {str(e)}</p>
        </div>
"""
        
        html += """
        <div class="result-card">
            <h3>📊 Summary</h3>
            <p>Analysis completed on manufacturing images using ErgonomicXAI pipeline.</p>
            <p><strong>Key Features:</strong></p>
            <ul>
                <li>REBA (Rapid Entire Body Assessment) scoring</li>
                <li>Body part risk breakdown (trunk, arms, legs)</li>
                <li>AI-powered explainable risk attribution</li>
                <li>Actionable ergonomic recommendations</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
        return html

def start_server(port=8080):
    """Start a simple HTTP server to view results"""
    os.chdir(Path(__file__).parent)
    
    with socketserver.TCPServer(("", port), ResultsHandler) as httpd:
        print(f"🌐 ErgonomicXAI Results Viewer")
        print(f"📊 Server running at: http://localhost:{port}")
        print(f"🔗 Open in browser: http://localhost:{port}")
        print("Press Ctrl+C to stop the server")
        
        try:
            webbrowser.open(f"http://localhost:{port}")
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

if __name__ == "__main__":
    start_server()
