const http = require('http');
const { WebSocketServer } = require('ws');

const port = process.env.PORT || 8080;

// Create an HTTP server so Render health checks can pass
const server = http.createServer((req, res) => {
    if (req.url === '/' || req.url === '/healthz') {
        res.writeHead(200, { 'Content-Type': 'text/plain' });
        res.end('Smart Surveillance Cloud WebSocket Broker is Active and Running!');
    } else {
        res.writeHead(404);
        res.end();
    }
});

// Bind WebSocket Server to the same HTTP server port
const wss = new WebSocketServer({ server });

let backendClient = null;
const frontendClients = new Set();

wss.on('connection', (ws, req) => {
    // Parse role query parameter (role=backend or role=frontend)
    const urlParams = new URLSearchParams(req.url.split('?')[1]);
    const role = urlParams.get('role');

    if (role === 'backend') {
        if (backendClient) {
            console.log("[Broker] Replacing existing backend connection.");
            backendClient.close();
        }
        backendClient = ws;
        console.log("[Broker] Surveillance Backend connected.");

        ws.on('message', (message) => {
            const msgStr = message.toString();
            // Relay telemetry to all connected frontend clients
            frontendClients.forEach(client => {
                if (client.readyState === 1) {
                    client.send(msgStr);
                }
            });
        });

        ws.on('close', () => {
            console.log("[Broker] Surveillance Backend disconnected.");
            if (backendClient === ws) {
                backendClient = null;
            }
        });

        ws.on('error', (err) => {
            console.error("[Broker] Backend connection error:", err);
        });
    } else {
        frontendClients.add(ws);
        console.log(`[Broker] Viewer connected. Total viewers: ${frontendClients.size}`);

        ws.on('message', (message) => {
            const msgStr = message.toString();
            // Relay viewer control signals (PAUSE, RESUME, STOP) to the backend
            if (backendClient && backendClient.readyState === 1) {
                backendClient.send(msgStr);
            }
        });

        ws.on('close', () => {
            frontendClients.delete(ws);
            console.log(`[Broker] Viewer disconnected. Total viewers: ${frontendClients.size}`);
        });

        ws.on('error', (err) => {
            console.error("[Broker] Viewer connection error:", err);
        });
    }
});

server.listen(port, () => {
    console.log(`[Broker] Server running on port ${port}`);
});
