# BetaBuddy payment Worker

`GET /health` reports service availability. `POST /v1/entitlement` accepts a
StoreKit 2 `signedTransactionInfo` from the app, verifies Apple's certificate
chain and signature, then asks Apple's App Store Server API for the current
transaction state. It grants the lifetime entitlement only for
`com.betabuddy.ios.lifetime` when the transaction has not been revoked.

The Worker never receives videos, pose data, route choices, or payment card
details. It does receive the signed App Store transaction for verification.
Do not log, persist, or put that token in a URL. The 1.2.0 app does not call
this endpoint; it uses StoreKit's on-device verified entitlement. This endpoint
does not yet deliver premium gym packs or training recommendations.

The Worker code and `pay.dozycat.ai` production domain have been deployed.
`GET /health` and rejection of an invalid signed transaction passed. Live
transaction verification still requires the App Store Connect **In-App Purchase**
API key. Set its issuer ID and key ID as
Worker secrets `APPLE_ISSUER_ID` and `APPLE_KEY_ID`, and the `.p8` private key
as `APPLE_PRIVATE_KEY`. Never commit the private key. The deployed Worker uses
the `nodejs_compat` flag. For future code updates, run `bun run check` and
`bun run deploy`; `wrangler.jsonc` routes the Worker to `pay.dozycat.ai`.

The app ID, bundle ID, and product ID are fixed in `src/index.ts`. Apple's
public root certificates in `src/apple-roots.ts` came from
<https://www.apple.com/certificateauthority/>.
