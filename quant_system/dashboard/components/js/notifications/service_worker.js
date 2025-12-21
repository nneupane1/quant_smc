// service_worker.js

self.addEventListener("notificationclick", function (e) {
    e.notification.close();
    e.waitUntil(
        clients.openWindow("/")
    );
});
