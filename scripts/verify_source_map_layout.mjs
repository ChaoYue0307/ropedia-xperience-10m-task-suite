import http from "node:http";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const docsRoot = path.join(repoRoot, "docs");
const outputDir = path.join(repoRoot, "tmp", "website-checks");

const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".mp4": "video/mp4"
};

function safePath(urlPath) {
  const decoded = decodeURIComponent(urlPath.split("?")[0]);
  const relative = decoded === "/" ? "index.html" : decoded.replace(/^\/+/, "");
  const candidate = path.resolve(docsRoot, relative);
  if (!candidate.startsWith(docsRoot)) return null;
  return candidate;
}

function createServer() {
  return http.createServer(async (request, response) => {
    const filePath = safePath(request.url || "/");
    if (!filePath) {
      response.writeHead(403);
      response.end("Forbidden");
      return;
    }
    try {
      const body = await fs.readFile(filePath);
      response.writeHead(200, { "content-type": contentTypes[path.extname(filePath)] || "application/octet-stream" });
      response.end(body);
    } catch {
      response.writeHead(404);
      response.end("Not found");
    }
  });
}

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolve(server.address().port));
  });
}

async function inspectViewport(page, baseUrl, viewport, name) {
  await page.setViewportSize(viewport);
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.locator("#features").scrollIntoViewIfNeeded();

  const metrics = await page.evaluate(() => {
    const rectFor = (selector) => {
      const element = document.querySelector(selector);
      if (!element) return null;
      const rect = element.getBoundingClientRect();
      return {
        selector,
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
        top: rect.top,
        bottom: rect.bottom,
        left: rect.left,
        right: rect.right
      };
    };
    const section = rectFor("#features");
    const head = rectFor("#features .section-head");
    const grid = rectFor("#features .source-map-grid");
    const panel = rectFor("#features .source-map-panel");
    const chartFrameElement = document.querySelector("#features .source-map-chart-frame");
    const chartFrame = rectFor("#features .source-map-chart-frame");
    const cards = [...document.querySelectorAll("#features .source-map-card")].map((element) => {
      const rect = element.getBoundingClientRect();
      return { width: rect.width, height: rect.height, top: rect.top, bottom: rect.bottom };
    });
    const legend = [...document.querySelectorAll("#features .source-map-legend article")].map((element) => {
      const rect = element.getBoundingClientRect();
      return { width: rect.width, height: rect.height, top: rect.top, bottom: rect.bottom };
    });
    const verticalOrderOk = Boolean(head && grid && panel && head.bottom <= grid.top + 1 && grid.bottom <= panel.top + 1);
    const bodyOverflow = document.documentElement.scrollWidth - document.documentElement.clientWidth;
    return {
      bodyOverflow,
      section,
      head,
      grid,
      panel,
      chartFrame: chartFrame && {
        ...chartFrame,
        clientWidth: chartFrameElement.clientWidth,
        scrollWidth: chartFrameElement.scrollWidth
      },
      cardCount: cards.length,
      cards,
      legendCount: legend.length,
      legend,
      verticalOrderOk
    };
  });

  const failures = [];
  if (metrics.cardCount !== 3) failures.push(`expected 3 source-map cards, found ${metrics.cardCount}`);
  if (metrics.legendCount !== 3) failures.push(`expected 3 legend items, found ${metrics.legendCount}`);
  if (!metrics.verticalOrderOk) failures.push("section head, source cards, and chart panel overlap or are out of order");
  if (metrics.bodyOverflow > 2) failures.push(`page has horizontal overflow of ${metrics.bodyOverflow}px`);
  if (!metrics.chartFrame || metrics.chartFrame.width <= 0 || metrics.chartFrame.clientWidth <= 0) failures.push("chart frame is not visible");
  if (metrics.cards.some((card) => card.width < 120 || card.height < 80)) failures.push("one or more source-map cards collapsed");
  if (metrics.legend.some((item) => item.width < 120 || item.height < 50)) failures.push("one or more legend items collapsed");

  const screenshotPath = path.join(outputDir, `source-map-${name}.png`);
  await page.locator("#features").screenshot({ path: screenshotPath });
  return { name, viewport, screenshotPath, metrics, failures };
}

async function main() {
  await fs.mkdir(outputDir, { recursive: true });
  const server = createServer();
  const port = await listen(server);
  const baseUrl = `http://127.0.0.1:${port}/index.html`;
  const browser = await chromium.launch({ headless: true });

  try {
    const page = await browser.newPage();
    const results = [
      await inspectViewport(page, baseUrl, { width: 1440, height: 1100 }, "desktop"),
      await inspectViewport(page, baseUrl, { width: 390, height: 1200 }, "mobile")
    ];
    const failures = results.flatMap((result) => result.failures.map((failure) => `${result.name}: ${failure}`));
    console.log(JSON.stringify({ baseUrl, results, failures }, null, 2));
    if (failures.length) process.exitCode = 1;
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
