"""Lab 04 - Capture High-Resolution Screenshots of the Streamlit Web Application using Chrome DevTools Protocol."""
import base64
import json
import os
import subprocess
import time
import urllib.request
import websocket

EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def send_cdp(ws, req_id, method, params=None):
    payload = {"id": req_id, "method": method, "params": params or {}}
    ws.send(json.dumps(payload))
    while True:
        resp = json.loads(ws.recv())
        if resp.get("id") == req_id:
            return resp.get("result", {})


def eval_js(ws, req_id, expr):
    res = send_cdp(ws, req_id, "Runtime.evaluate", {"expression": expr, "returnByValue": True})
    return res.get("result", {}).get("value")


def capture_screenshots():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Launch Edge with remote debugging port
    edge_proc = subprocess.Popen([
        EDGE_EXE,
        "--headless=new",
        "--remote-debugging-port=9222",
        "--remote-allow-origins=*",
        "--no-sandbox",
        "--disable-gpu",
        "--window-size=1400,1050",
    ])
    time.sleep(2)

    try:
        # Get target list from CDP
        with urllib.request.urlopen("http://localhost:9222/json", timeout=10) as r:
            targets = json.loads(r.read().decode())
        page_target = [t for t in targets if t.get("type") == "page"][0]
        ws_url = page_target["webSocketDebuggerUrl"]

        ws = websocket.create_connection(ws_url, timeout=30)
        req_id = 1

        # Set viewport
        send_cdp(ws, req_id, "Emulation.setDeviceMetricsOverride", {
            "width": 1400,
            "height": 1050,
            "deviceScaleFactor": 1,
            "mobile": False,
        })
        req_id += 1

        # Navigate to Streamlit
        send_cdp(ws, req_id, "Page.navigate", {"url": "http://localhost:8501"})
        req_id += 1

        # Wait for Streamlit to hydrate and finish running
        print("Waiting for Streamlit app to load and finish running...")
        start_t = time.time()
        while time.time() - start_t < 35:
            ready = eval_js(
                ws, req_id,
                "document.querySelector('button[kind=\"primary\"]') !== null && "
                "document.querySelector('[data-testid=\"stMarkdownContainer\"]') !== null && "
                "!document.querySelector('[data-testid=\"stStatusWidget\"]') && "
                "!document.body.innerText.includes('Connecting')"
            )
            req_id += 1
            if ready:
                print("Streamlit app fully loaded and initial calculation complete!")
                break
            time.sleep(1)

        time.sleep(3)  # Let plots finish rendering

        # 1. Capture Screenshot 1: Tab 1 (Classifier, Score, Threshold & Per-Feature Explanations)
        res1 = send_cdp(ws, req_id, "Page.captureScreenshot", {"format": "png"})
        req_id += 1
        img_data1 = base64.b64decode(res1["data"])
        path1 = os.path.join(RESULTS_DIR, "app_screenshot_1_classify.png")
        with open(path1, "wb") as f:
            f.write(img_data1)
        print(f"Saved: {path1} ({len(img_data1)} bytes)")

        # 2. Click Tab 2 (Threshold Policy, Calibration & Metrics)
        eval_js(
            ws, req_id,
            "(() => {"
            "  const tabs = document.querySelectorAll('button[data-baseweb=\"tab\"]');"
            "  if (tabs.length >= 2) { tabs[1].click(); return true; }"
            "  return false;"
            "})()"
        )
        req_id += 1
        time.sleep(3)

        # Capture Screenshot 2: Tab 2 (Results Table, Threshold Trade-Off & Reliability Diagram)
        res2 = send_cdp(ws, req_id, "Page.captureScreenshot", {"format": "png"})
        req_id += 1
        img_data2 = base64.b64decode(res2["data"])
        path2 = os.path.join(RESULTS_DIR, "app_screenshot_2_evaluation.png")
        with open(path2, "wb") as f:
            f.write(img_data2)
        print(f"Saved: {path2} ({len(img_data2)} bytes)")

        # 3. Click Tab 3 (Data Augmentation & Real UCI Dataset)
        eval_js(
            ws, req_id,
            "(() => {"
            "  const tabs = document.querySelectorAll('button[data-baseweb=\"tab\"]');"
            "  if (tabs.length >= 3) { tabs[2].click(); return true; }"
            "  return false;"
            "})()"
        )
        req_id += 1
        time.sleep(3)

        # Capture Screenshot 3: Tab 3 (Data Augmentation Breakdown & Benchmarks)
        res3 = send_cdp(ws, req_id, "Page.captureScreenshot", {"format": "png"})
        req_id += 1
        img_data3 = base64.b64decode(res3["data"])
        path3 = os.path.join(RESULTS_DIR, "app_screenshot_3_augmentation.png")
        with open(path3, "wb") as f:
            f.write(img_data3)
        print(f"Saved: {path3} ({len(img_data3)} bytes)")

        ws.close()
    finally:
        edge_proc.terminate()


if __name__ == "__main__":
    capture_screenshots()
