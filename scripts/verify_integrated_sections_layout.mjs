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
  await page.goto(`${baseUrl}#takeaways`, { waitUntil: "networkidle" });
  await page.waitForSelector("#takeaways #result-matrix-table", { timeout: 20000 });
  await page.waitForSelector("#resultScoreTable tbody tr:nth-child(2)", { state: "attached", timeout: 20000 });
  await page.locator("#takeaways").scrollIntoViewIfNeeded();
  await page.waitForTimeout(500);

  const metrics = await page.evaluate(() => {
    const rectFor = (selector) => {
      const element = document.querySelector(selector);
      if (!element) return null;
      const rect = element.getBoundingClientRect();
      return {
        selector,
        width: rect.width,
        height: rect.height,
        top: rect.top,
        bottom: rect.bottom,
        left: rect.left,
        right: rect.right
      };
    };
    const oldStandalone = ["models", "neural", "diagnostics", "extensions"].filter((id) => (
      Boolean(document.querySelector(`main > section#${id}`))
    ));
    const anchors = {
      modelsParent: document.querySelector("#models")?.closest("section")?.id || null,
      neuralParent: document.querySelector("#neural")?.closest("section")?.id || null,
      diagnosticsParent: document.querySelector("#diagnostics")?.closest("section")?.id || null,
      extensionsParent: document.querySelector("#extensions")?.closest("section")?.id || null
    };
    const resultJumpLinks = [...document.querySelectorAll(".result-jump-row a")].map((link) => link.getAttribute("href"));
    const directionJumpLinks = [...document.querySelectorAll(".direction-jump-row a")].map((link) => link.getAttribute("href"));
    const taskAxisRuleCards = [...document.querySelectorAll(".task-axis-reader-rule article")].map((card) => (
      card.textContent.replace(/\s+/g, " ").trim()
    ));
    const taskAxisSummaryCounts = [...document.querySelectorAll(".task-axis-summary-card .task-axis-count")].map((node) => (
      node.textContent.trim()
    ));
    const resultMatrixDetails = document.querySelector(".result-matrix-details");
    const resultScoreRows = document.querySelectorAll("#resultScoreTable tbody tr").length;
    const resultBlocks = [...document.querySelectorAll("#takeaways .result-subsection")].map((block) => ({
      id: block.id,
      rect: rectFor(`#${block.id}`),
      text: block.textContent.replace(/\s+/g, " ").trim().slice(0, 160)
    }));
    const directionBlock = rectFor("#extensions");
    return {
      bodyOverflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      takeaways: rectFor("#takeaways"),
      directions: rectFor("#directions"),
      resultMatrix: rectFor("#result-matrix-table"),
      directionBlock,
      oldStandalone,
      anchors,
      resultJumpLinks,
      directionJumpLinks,
      taskAxisRuleCards,
      taskAxisSummaryCounts,
      resultMatrixDetails: resultMatrixDetails ? { open: resultMatrixDetails.open } : null,
      resultScoreRows,
      resultBlocks,
      sectionCount: document.querySelectorAll("main > section[data-project-tab]").length
    };
  });

  const failures = [];
  if (metrics.oldStandalone.length) failures.push(`old standalone sections remain: ${metrics.oldStandalone.join(", ")}`);
  if (metrics.anchors.modelsParent !== "takeaways") failures.push("#models is not inside #takeaways");
  if (metrics.anchors.neuralParent !== "takeaways") failures.push("#neural is not inside #takeaways");
  if (metrics.anchors.diagnosticsParent !== "takeaways") failures.push("#diagnostics is not inside #takeaways");
  if (metrics.anchors.extensionsParent !== "directions") failures.push("#extensions is not inside #directions");
  for (const href of ["#result-lines", "#result-matrix-table", "#models", "#neural", "#diagnostics"]) {
    if (!metrics.resultJumpLinks.includes(href)) failures.push(`missing Results quick link ${href}`);
  }
  for (const href of ["#direction-coverage", "#direction-baselines", "#extensions", "#suite"]) {
    if (!metrics.directionJumpLinks.includes(href)) failures.push(`missing Directions quick link ${href}`);
  }
  for (const count of ["20", "4", "3"]) {
    if (!metrics.taskAxisSummaryCounts.includes(count)) failures.push(`missing public structure count ${count}`);
  }
  if (metrics.taskAxisRuleCards.length !== 3) failures.push(`expected 3 public-structure reader-rule cards, found ${metrics.taskAxisRuleCards.length}`);
  for (const marker of ["20-task layer", "4-direction layer", "3-pipeline layer"]) {
    if (!metrics.taskAxisRuleCards.some((text) => text.includes(marker))) failures.push(`missing reader-rule marker: ${marker}`);
  }
  if (metrics.resultBlocks.length !== 3) failures.push(`expected 3 integrated result subsections, found ${metrics.resultBlocks.length}`);
  if (!metrics.resultMatrixDetails) failures.push("result matrix disclosure is missing");
  if (metrics.resultMatrixDetails?.open) failures.push("result matrix disclosure should be closed by default");
  if (metrics.resultScoreRows < 2) failures.push(`result matrix table did not populate enough rows: ${metrics.resultScoreRows}`);
  if (metrics.resultBlocks.some((block) => !block.rect || block.rect.width < 260 || block.rect.height < 160)) {
    failures.push("one or more result subsections collapsed below readable size");
  }
  if (!metrics.directionBlock || metrics.directionBlock.width < 260 || metrics.directionBlock.height < 260) {
    failures.push("direction provenance subsection collapsed below readable size");
  }
  if (metrics.bodyOverflow > 2) failures.push(`page has horizontal overflow of ${metrics.bodyOverflow}px`);

  const screenshotPath = path.join(outputDir, `integrated-results-${name}.png`);
  const directionScreenshotPath = path.join(outputDir, `integrated-directions-${name}.png`);
  await page.locator("#takeaways").screenshot({ path: screenshotPath });
  await page.goto(`${baseUrl}#extensions`, { waitUntil: "networkidle" });
  await page.waitForSelector("#directions #extensions", { timeout: 20000 });
  await page.locator("#directions").screenshot({ path: directionScreenshotPath });

  return { name, viewport, screenshotPath, directionScreenshotPath, metrics, failures };
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
