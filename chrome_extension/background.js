// background.js — service worker (MV3)
// Handles install events and cross-tab messaging if needed.

chrome.runtime.onInstalled.addListener(() => {
  console.log("SentimentScope installed.");
});
