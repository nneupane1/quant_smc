// notify.js
// Desktop + mobile push + audio alerts.

window.Notifier = (function () {

    const sound = new Audio("/assets/sounds/alert_chime.mp3");

    function requestPermission() {
        if (Notification.permission !== "granted") {
            Notification.requestPermission();
        }
    }

    function desktopNotify(title, body) {
        if (Notification.permission === "granted") {
            new Notification(title, { body: body });
        }
    }

    function popupNotify(title, body, level) {
        const wrap = document.getElementById("notif-container");
        const div = document.createElement("div");
        div.className = `notif notif-${level}`;

        div.innerHTML = `
            <div class="notif-title">${title}</div>
            <div class="notif-body">${body}</div>
        `;

        wrap.prepend(div);
        setTimeout(() => div.classList.add("show"), 30);
        setTimeout(() => {
            div.classList.remove("show");
            setTimeout(() => wrap.removeChild(div), 300);
        }, 6000);
    }

    // Real mobile push (PWA + service worker)
    function mobilePush(title, body) {
        if (navigator.serviceWorker) {
            navigator.serviceWorker.ready.then(reg => {
                reg.showNotification(title, { body: body });
            });
        }
    }

    function playSound() {
        sound.currentTime = 0;
        sound.play();
    }

    function dispatch(event) {
        if (event.type !== "notify") return;

        const p = event.payload;

        requestPermission();
        desktopNotify(p.title, p.body);
        mobilePush(p.title, p.body);
        popupNotify(p.title, p.body, p.level);
        playSound();
    }

    return { dispatch };

})();
