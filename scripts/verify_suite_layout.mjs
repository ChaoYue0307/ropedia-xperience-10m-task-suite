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
  await page.goto(`${baseUrl}#suite`, { waitUntil: "networkidle" });
  await page.waitForSelector("#taskGrid .task-card", { timeout: 20000 });
  await page.locator("#tasks").scrollIntoViewIfNeeded();
  await page.waitForTimeout(500);

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
    const suite = rectFor("#suite");
    const map = rectFor("#task-suite-map");
    const radars = rectFor("#suite-radars");
    const tasks = rectFor("#tasks");
    const taskGrid = rectFor("#taskGrid");
    const jumpLinks = [...document.querySelectorAll(".suite-jump-row a")].map((link) => link.getAttribute("href"));
    const taskCards = [...document.querySelectorAll("#taskGrid .task-card")].map((card) => {
      const rect = card.getBoundingClientRect();
      const style = window.getComputedStyle(card);
      const methodScoreCount = card.querySelectorAll(".task-method-score").length;
      return {
        width: rect.width,
        height: rect.height,
        opacity: Number(style.opacity),
        methodScoreCount,
        text: card.textContent.replace(/\s+/g, " ").trim().slice(0, 160)
      };
    });
    const filters = [...document.querySelectorAll("#tasks .filter")].map((button) => button.textContent.trim());
    const oldStandaloneTasks = document.querySelector("main > section#tasks");
    const suiteText = document.querySelector("#suite .section-head")?.textContent?.replace(/\s+/g, " ").trim() || "";
    const bodyOverflow = document.documentElement.scrollWidth - document.documentElement.clientWidth;
    return {
      bodyOverflow,
      suite,
      map,
      radars,
      tasks,
      taskGrid,
      jumpLinks,
      taskCardCount: taskCards.length,
      methodScoreCounts: taskCards.map((card) => card.methodScoreCount),
      collapsedTaskCards: taskCards.filter((card) => card.width < 220 || card.height < 180),
      incompleteMethodScoreCards: taskCards.filter((card) => card.methodScoreCount !== 9),
      hiddenTaskCards: taskCards.filter((card) => card.opacity < 0.8),
      filters,
      hasOldStandaloneTasks: Boolean(oldStandaloneTasks),
      suiteText
    };
  });

  const failures = [];
  if (!metrics.suite) failures.push("missing #suite section");
  if (!metrics.map) failures.push("missing task-suite map");
  if (!metrics.radars) failures.push("missing integrated radar block");
  if (!metrics.tasks) failures.push("missing #tasks task-card anchor inside suite");
  if (metrics.hasOldStandaloneTasks) failures.push("old standalone main > section#tasks still exists");
  if (metrics.taskCardCount !== 20) failures.push(`expected 20 task cards, found ${metrics.taskCardCount}`);
  if (metrics.incompleteMethodScoreCards.length) failures.push(`${metrics.incompleteMethodScoreCards.length} task cards do not show all 9 method scores`);
  if (metrics.filters.length !== 5) failures.push(`expected 5 task filters, found ${metrics.filters.length}`);
  if (!metrics.jumpLinks.includes("#task-suite-map") || !metrics.jumpLinks.includes("#suite-radars") || !metrics.jumpLinks.includes("#tasks")) {
    failures.push("suite jump row is missing required anchors");
  }
  if (!metrics.suiteText.includes("Task map, radar comparisons, task cards")) {
    failures.push("suite heading copy no longer explains the integrated reading flow");
  }
  if (metrics.bodyOverflow > 2) failures.push(`page has horizontal overflow of ${metrics.bodyOverflow}px`);
  if (metrics.map && metrics.radars && metrics.map.bottom > metrics.radars.top + 1) failures.push("task map overlaps radar block");
  if (metrics.radars && metrics.tasks && metrics.radars.bottom > metrics.tasks.top + 1) failures.push("radar block overlaps task cards");
  if (metrics.collapsedTaskCards.length) failures.push(`${metrics.collapsedTaskCards.length} task cards collapsed below readable size`);
  if (metrics.hiddenTaskCards.length) failures.push(`${metrics.hiddenTaskCards.length} task cards stayed hidden after scrolling to the task-card block`);

  const screenshotPath = path.join(outputDir, `suite-integrated-${name}.png`);
  const taskScreenshotPath = path.join(outputDir, `suite-task-cards-${name}.png`);
  await page.locator("#suite").screenshot({ path: screenshotPath });
  await page.locator("#tasks").screenshot({ path: taskScreenshotPath });
  return { name, viewport, screenshotPath, taskScreenshotPath, metrics, failures };
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
      await inspectViewport(page, baseUrl, { width: 1440, height: 1600 }, "desktop"),
      await inspectViewport(page, baseUrl, { width: 390, height: 1400 }, "mobile")
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
