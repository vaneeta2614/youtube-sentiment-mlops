/**
 * content.js — Injected on youtube.com/watch pages
 * Reads the current video ID and reports it to popup on request.
 */
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "GET_VIDEO_ID") {
    const params = new URLSearchParams(window.location.search);
    sendResponse({ videoId: params.get("v") || null });
  }
});
