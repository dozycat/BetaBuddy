# BetaBuddy payment Worker

A Cloudflare Worker at `pay.dozycat.ai`. The app does not call it: the lifetime
entitlement is checked on the device with StoreKit's verified transactions.

- `GET /health` reports service availability.
- `POST /v1/entitlement` accepts a StoreKit 2 `signedTransactionInfo`, verifies
  Apple's certificate chain and signature, then asks Apple's App Store Server API
  for the current transaction state. It reports the lifetime entitlement as
  active only for `com.betabuddy.ios.lifetime` when the transaction has not been
  revoked.

The Worker never receives videos, pose data, route choices or payment card
details. It does receive the signed App Store transaction for verification:
never log it, persist it or put it in a URL.

The App Store Server API key is held as the Worker secrets `APPLE_ISSUER_ID`,
`APPLE_KEY_ID` and `APPLE_PRIVATE_KEY`; never commit the private key. The
App's Apple ID (needed to verify production transactions) is the secret
`APP_APPLE_ID`; it is account information and is not written in the repo. The
deployed Worker predates this secret: set it with
`bunx wrangler secret put APP_APPLE_ID` before the next deploy. The bundle
ID and product ID are set in `src/index.ts`. Apple's public root certificates in
`src/apple-roots.ts` come from <https://www.apple.com/certificateauthority/>.

```sh
bun run check     # type-check
bun run deploy    # wrangler deploy; wrangler.jsonc routes the Worker to pay.dozycat.ai
```

Not yet verified: a real sandbox purchase through this endpoint.
