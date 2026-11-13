# BetaBuddy support site

English support and privacy pages at <https://betabuddy.dozycat.ai>, deployed as a Cloudflare static Worker (`betabuddy-support`). The `public/` files are generated from `src/index.ts` with `bun build.ts`. Cloudflare's dashboard currently hosts a direct upload of those files; `wrangler.jsonc` is ready for later CLI deployment to the same Worker.

The App Store support URL is `/`, the privacy policy URL is `/privacy`, and the in-app About screen points to both. Update this site and the bundled privacy text together when data handling changes.
