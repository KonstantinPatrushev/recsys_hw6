// Descriptive statistics of the report users' Top-10 lists (readme 7.3 and 8): overlap between
// the two users within each method, overlap of PPR with the other methods, mean training
// popularity, the one-factor demonstration, and the users' training histories.
import vm from 'vm';
import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const page = JSON.parse(readFileSync(join(root, 'evidence', process.argv[2] || 'run_full.json'), 'utf8'));
const ind = JSON.parse(readFileSync(join(root, 'evidence', 'independent.json'), 'utf8'));
const ctx = vm.createContext({ console, fetch: async (p) => ({ ok: true, text: async () => readFileSync(join(root, p), 'latin1') }) });
vm.runInContext(readFileSync(join(root, 'data.js'), 'utf8'), ctx);
await vm.runInContext('loadData()', ctx);
const { movies, split } = vm.runInContext('({ movies, split: splitByTimestamp() })', ctx);
const byId = new Map(movies.map(m => [m.id, m]));

const lists = {};
for (const u of ['655', '895']) {
    lists[u] = { ...page.lists[u], mf_hw5: ind.reportUsers[u].mf_hw5 };
}
const ids = (l) => l.map(x => x.movieId);
const common = (a, b) => ids(a).filter(id => ids(b).includes(id)).length;
const meanPop = (l) => l.reduce((s, x) => s + x.popularity, 0) / l.length;
const out = { withinMethod: {}, pprOverlap: {}, meanPopularity: {}, histories: {} };
for (const m of ['popularity', 'pagerank', 'ppr', 'mf', 'mf_hw5']) out.withinMethod[m] = common(lists['655'][m], lists['895'][m]);
for (const u of ['655', '895']) {
    out.pprOverlap[u] = Object.fromEntries(['popularity', 'pagerank', 'mf_hw5'].map(m => [m, common(lists[u].ppr, lists[u][m])]));
    out.meanPopularity[u] = Object.fromEntries(Object.keys(lists[u]).map(m => [m, +meanPop(lists[u][m]).toFixed(1)]));
    const uid = Number(u);
    const train = split.train.filter(r => r.userId === uid);
    const genres = new Map();
    for (const r of train.filter(r => r.rating >= 4)) for (const g of byId.get(r.movieId).genres) genres.set(g, (genres.get(g) || 0) + 1);
    out.histories[u] = {
        trainRatings: train.length, positive: train.filter(r => r.rating >= 4).length,
        topGenres: [...genres].sort((a, b) => b[1] - a[1]).slice(0, 4),
        fiveStar: train.filter(r => r.rating === 5).map(r => byId.get(r.movieId).title).slice(0, 10),
        years: [...new Set(train.map(r => new Date(r.timestamp * 1000).toISOString().slice(0, 7)))].sort()
    };
}
const demo = page.reportDemo;
out.demo = {
    settings: demo.map(s => ({ label: s.label, user: s.user, alpha: s.alpha, meanPopularity: +meanPop(s.recs).toFixed(1) })),
    sharedWithBaseline: { changeSeed: common(demo[0].recs, demo[1].recs), changeAlpha: common(demo[0].recs, demo[2].recs) },
    changedByAlpha: { removed: demo[0].recs.filter(x => !ids(demo[2].recs).includes(x.movieId)).map(x => `${x.title} (pop ${x.popularity})`),
                      added: demo[2].recs.filter(x => !ids(demo[0].recs).includes(x.movieId)).map(x => `${x.title} (pop ${x.popularity})`) },
    pageDemoUsers: page.pageDemo.users, pageDemoCommon: page.pageDemo.common
};
writeFileSync(join(root, 'evidence', 'list_stats.json'), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));
