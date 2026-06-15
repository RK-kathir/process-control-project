// dashboard-disconnect.js
// ------------------------------------------------------------------
// Adds a manual Disconnect / Reconnect control + live status pill to
// the Tuning Bot dashboard.
//
// ASSUMPTIONS:
//   - A Socket.IO client instance named `socket` already exists
//     (created elsewhere in your dashboard JS), e.g.:
//
//       const socket = io(SERVER_URL, { transports: ['websocket'] });
//
//   - Your HTML contains two elements (add these wherever your
//     connection status currently lives):
//
//       <div class="connection-bar">
//         <span id="connection-status" class="status-pill">Connecting...</span>
//         <button id="disconnect-btn" class="disconnect-btn">Disconnect</button>
//       </div>
//
// USAGE:
//   Load this AFTER the script that creates `socket`:
//     <script src="dashboard-disconnect.js"></script>
// ------------------------------------------------------------------

(function () {
  const btn = document.getElementById('disconnect-btn');
  const statusPill = document.getElementById('connection-status');

  if (!btn || !statusPill) {
    console.warn('[DisconnectControl] #disconnect-btn or #connection-status not found in DOM.');
    return;
  }

  if (typeof socket === 'undefined') {
    console.warn('[DisconnectControl] Global `socket` instance not found.');
    return;
  }

  function setStatus(connected) {
    if (connected) {
      statusPill.textContent = 'Connected';
      statusPill.classList.remove('status-disconnected');
      statusPill.classList.add('status-connected');
      btn.textContent = 'Disconnect';
      btn.dataset.action = 'disconnect';
      btn.disabled = false;
    } else {
      statusPill.textContent = 'Disconnected';
      statusPill.classList.remove('status-connected');
      statusPill.classList.add('status-disconnected');
      btn.textContent = 'Reconnect';
      btn.dataset.action = 'reconnect';
      btn.disabled = false;
    }
  }

  // Reflect the socket's current state immediately on load
  setStatus(socket.connected);

  socket.on('connect', () => setStatus(true));
  socket.on('disconnect', () => setStatus(false));

  btn.addEventListener('click', () => {
    if (btn.dataset.action === 'disconnect') {
      console.log('[DisconnectControl] Manual disconnect requested.');
      btn.disabled = true; // briefly disable to avoid double clicks mid-transition
      socket.disconnect();
    } else {
      console.log('[DisconnectControl] Manual reconnect requested.');
      btn.disabled = true;
      socket.connect();
    }
    // re-enable after a short delay in case no connect/disconnect event fires
    setTimeout(() => { btn.disabled = false; }, 1500);
  });
})();
