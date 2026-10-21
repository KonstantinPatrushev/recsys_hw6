// Independent re-implementation in plain JavaScript, float64, sparse — shares no code with
// graph.js / script.js except the data layer (data.js: loadData + splitByTimestamp, which the
// assignment fixes). It recomputes the graph, global PageRank, PPR, popularity, the shared
// cohort and Recall@10, and evaluates the MF checkpoint selected in assignment 5
// (hw5/evidence/run_main.json) under the same protocol. If evidence/run_full.json (the page's
// own run) exists, the numbers and the report users' lists are compared with it.
import vm from 'vm';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const ctx = vm.createContext({ console, fetch: async (p) => ({ ok: true, text: async () => readFileSync(join(root, p), 'latin1') }) });
vm.runInContext(readFileSync(join(root, 'data.js'), 'utf8'), ctx);
await vm.runInContext('loadData()', ctx);
const { movies, ratings, split } = vm.runInContext('({ movies, ratings, split: splitByTimestamp() })', ctx);
const { train, val, test } = split;
const K = 10, TOL = 1e-6, MAX_ITER = 500, ALPHAS = [0.5, 0.7, 0.85, 0.95];
const REPORT_USERS = [655, 895];
const title = new Map(movies.map(m => [m.id, m.title]));

// ---- graph over training ids, positive edges = rating >= 4 ----
const userIds = [...new Set(train.map(r => r.userId))].sort((a, b) => a - b);
const movieIds = [...new Set(train.map(r => r.movieId))].sort((a, b) => a - b);
const nU = userIds.length, nM = movieIds.length, n = nU + nM;
const uIdx = new Map(userIds.map((id, k) => [id, k])), mIdx = new Map(movieIds.map((id, k) => [id, nU + k]));
const adj = Array.from({ length: n }, () => []);
const seen = new Set();
for (const r of train) {
    if (r.rating < 4) continue;
    const key = r.userId + ':' + r.movieId;
    if (seen.has(key)) continue;
    seen.add(key);
    adj[uIdx.get(r.userId)].push(mIdx.get(r.movieId));
    adj[mIdx.get(r.movieId)].push(uIdx.get(r.userId));
}
const deg = adj.map(a => a.length);

// p_next = alpha * (Q^T p + d v) + (1 - alpha) v, from p = v
function propagate(alpha, v) {
    let p = Float64Array.from(v);
    let it = 0, res = Infinity;
    while (it < MAX_ITER) {
        const next = new Float64Array(n);
        let d = 0;
        for (let i = 0; i < n; i++) {
            if (deg[i] === 0) { d += p[i]; continue; }
            const share = p[i] / deg[i];
            for (const j of adj[i]) next[j] += share;
        }
        res = 0;
        for (let j = 0; j < n; j++) {
            next[j] = alpha * (next[j] + d * v[j]) + (1 - alpha) * v[j];
            res += Math.abs(next[j] - p[j]);
        }
        p = next; it++;
        if (res < TOL) break;
    }
    return { p, iterations: it, residual: res, converged: res < TOL };
}

// ---- protocol ----
const candidates = new Set(movieIds);
const pop = new Map();
for (const r of train) if (r.rating >= 4) pop.set(r.movieId, (pop.get(r.movieId) || 0) + 1); // pairs are unique
const ratedIn = (rows) => { const m = new Map(); for (const r of rows) { if (!m.has(r.userId)) m.set(r.userId, new Set()); m.get(r.userId).add(r.movieId); } return m; };
const trainRated = ratedIn(train), valRated = ratedIn(val);
const exclusion = (splitName, u) => new Set([...(trainRated.get(u) || []), ...(splitName === 'test' ? (valRated.get(u) || []) : [])]);
const positiveUsers = new Set(train.filter(r => r.rating >= 4).map(r => r.userId));
function relevantFor(splitName) {
    const rows = splitName === 'val' ? val : test;
    const rel = new Map();
    for (const r of rows) {
        if (r.rating < 4 || !candidates.has(r.movieId) || !positiveUsers.has(r.userId) || !trainRated.has(r.userId)) continue;
        if (exclusion(splitName, r.userId).has(r.movieId)) continue;
        if (!rel.has(r.userId)) rel.set(r.userId, new Set());
        rel.get(r.userId).add(r.movieId);
    }
    return rel;
}
function topK(scoreOf, u, splitName) {
    const ex = exclusion(splitName, u);
    return movieIds.filter(id => !ex.has(id)).map(id => ({ id, s: scoreOf(id) }))
        .sort((a, b) => b.s - a.s || a.id - b.id).slice(0, K);
}
function recall(splitName, scorerFor) {
    const rel = relevantFor(splitName);
    const cohort = [...rel.keys()].sort((a, b) => a - b);
    let sum = 0;
    const lists = new Map();
    for (const u of cohort) {
        const list = topK(scorerFor(u), u, splitName);
        lists.set(u, list);
        sum += list.filter(x => rel.get(u).has(x.id)).length / rel.get(u).size;
    }
    return { recall: sum / cohort.length, cohort: cohort.length, lists };
}
const uniform = new Float64Array(n).fill(1 / n);
const oneHot = (u) => { const v = new Float64Array(n); v[uIdx.get(u)] = 1; return v; };
const pprScorer = (alpha) => (u) => { const { p } = propagate(alpha, oneHot(u)); return id => p[mIdx.get(id)]; };

// MF from assignment 5 (factors of its validation-selected checkpoint)
const hw5 = JSON.parse(readFileSync(join(root, '..', 'hw5', 'evidence', 'run_main.json'), 'utf8'));
const F = hw5.factors.rows[0][1], P = hw5.factors.user, Q = hw5.factors.item, mu = hw5.factors.globalMean;
const mfScorer = (u) => (id) => { let s = mu; for (let f = 0; f < F; f++) s += P[u * F + f] * Q[id * F + f]; return s; };

const out = { graph: { nUsers: nU, nMovies: nM, nodes: n, positiveEdges: seen.size,
    zeroDegreeUsers: userIds.filter(u => deg[uIdx.get(u)] === 0).length,
    zeroDegreeMovies: movieIds.filter(m => deg[mIdx.get(m)] === 0).length },
    mfCheckpoint: { source: 'hw5/evidence/run_main.json', epoch: hw5.best.epoch, valRMSE: hw5.best.valRMSE } };

out.sweep = ALPHAS.map(a => ({ alpha: a, recall: recall('val', pprScorer(a)).recall }));
let best = out.sweep[0];
for (const row of out.sweep) if (row.recall > best.recall) best = row; // ties keep the smaller alpha
out.selectedAlpha = best.alpha;

const prGlobal = propagate(best.alpha, uniform);
const methods = {
    popularity: () => (id) => pop.get(id) || 0,
    pagerank: () => (id) => prGlobal.p[mIdx.get(id)],
    ppr: pprScorer(best.alpha),
    mf_hw5: mfScorer,
};
out.test = {};
const testLists = {};
for (const [name, scorer] of Object.entries(methods)) {
    const r = recall('test', scorer);
    testLists[name] = r.lists;
    let popSum = 0;
    for (const list of r.lists.values()) popSum += list.reduce((s, x) => s + (pop.get(x.id) || 0), 0) / list.length;
    out.test[name] = { recall: r.recall, cohort: r.cohort, meanListPopularity: popSum / r.lists.size };
}
out.reportUsers = {};
for (const u of REPORT_USERS) {
    out.reportUsers[u] = {};
    for (const name of Object.keys(methods)) {
        out.reportUsers[u][name] = testLists[name].get(u).map(x => ({ movieId: x.id, title: title.get(x.id), score: x.s, popularity: pop.get(x.id) || 0 }));
    }
}

// ---- comparison with the page's own run ----
const pagePath = join(root, 'evidence', 'run_full.json');
let page = null;
try { page = existsSync(pagePath) ? JSON.parse(readFileSync(pagePath, 'utf8')) : null; } catch { page = null; } // still being written
if (page && page.comparison) {
    const pageRecall = Object.fromEntries((page.comparison || []).map(r => [r.method, r.recall]));
    out.vsPage = {
        selectedAlpha: { page: page.alpha, independent: best.alpha },
        sweep: (page.sweep?.rows || []).map((r, k) => ({ alpha: r.alpha, page: r.recall, independent: out.sweep[k]?.recall })),
        testRecall: ['popularity', 'pagerank', 'ppr'].map(m => ({ method: m, page: pageRecall[m], independent: out.test[m].recall })),
        reportUserLists: REPORT_USERS.flatMap(u => ['popularity', 'pagerank', 'ppr'].map(m => ({
            user: u, method: m,
            sameIds: JSON.stringify((page.lists?.[u]?.[m] || []).map(x => x.movieId)) === JSON.stringify(out.reportUsers[u][m].map(x => x.movieId)),
            maxScoreDiff: Math.max(0, ...(page.lists?.[u]?.[m] || []).map((x, k) => Math.abs(x.score - out.reportUsers[u][m][k].score)))
        })))
    };
}
writeFileSync(join(root, 'evidence', 'independent.json'), JSON.stringify(out, null, 1));
console.log(JSON.stringify({ graph: out.graph, sweep: out.sweep, selectedAlpha: out.selectedAlpha, test: out.test, vsPage: out.vsPage }, null, 1));
