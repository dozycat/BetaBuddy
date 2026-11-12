# BetaBuddy payment Worker

`GET /health` reports service availability. `POST /v1/entitlement` accepts a
StoreKit 2 `signedTransactionInfo` from the app, verifies Apple's certificate
chain and signature, then asks Apple's App Store Server API for the current
transaction state. It grants the lifetime entitlement only for
`com.betabuddy.ios.lifetime` when the transaction has not been revoked.

The Worker never receives videos, pose data, route choices, or payment card
details. It does receive the signed App Store transaction for verification.
Do not log, persist, or put that token in a URL. This endpoint does not yet
deliver premium gym packs or training recommendations.

Before deployment, activate the `dozycat.ai` Cloudflare zone and create an
App Store Connect **In-App Purchase** API key. Set its issuer ID and key ID as
Worker secrets `APPLE_ISSUER_ID` and `APPLE_KEY_ID`, and the `.p8` private key
as `APPLE_PRIVATE_KEY`. Never commit the private key. Then run `bun run check`
and `bun run deploy`. `wrangler.jsonc` routes the Worker to `pay.dozycat.ai`.

The app ID, bundle ID, and product ID are fixed in `src/index.ts`. Apple's
public root certificates in `src/apple-roots.ts` came from
<https://www.apple.com/certificateauthority/>.
