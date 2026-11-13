import { Buffer } from "node:buffer";
import { appleRootG2, appleRootG3 } from "./apple-roots";
import type { Environment as AppleEnvironment } from "@apple/app-store-server-library";

type Env = {
  APPLE_ISSUER_ID: string;
  APPLE_KEY_ID: string;
  APPLE_PRIVATE_KEY: string;
};

const bundleID = "com.betabuddy.ios";
const appAppleID = 6809249574;
const productID = "com.betabuddy.ios.lifetime";

function json(body: object, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/health") {
      return json({ service: "betabuddy-pay", status: "ok" });
    }
    if (request.method !== "POST" || url.pathname !== "/v1/entitlement") {
      return json({ error: "not_found" }, 404);
    }
    if (Number(request.headers.get("content-length") ?? 0) > 16_384) {
      return json({ error: "invalid_request" }, 400);
    }

    let signed: unknown;
    try {
      const body = await request.text();
      if (body.length > 16_384) return json({ error: "invalid_request" }, 400);
      signed = JSON.parse(body).signedTransactionInfo;
    } catch {
      return json({ error: "invalid_request" }, 400);
    }
    if (typeof signed !== "string" || signed.length > 16_000 || signed.split(".").length !== 3) {
      return json({ error: "invalid_request" }, 400);
    }
    // Loading Apple's library inside a request is required in Workers: one of its
    // dependencies initializes random state at import time.
    const Apple = await import("@apple/app-store-server-library");
    const roots = [Buffer.from(appleRootG2, "base64"), Buffer.from(appleRootG3, "base64")];
    let device: {
      environment: AppleEnvironment;
      checked: InstanceType<typeof Apple.SignedDataVerifier>;
      transaction: Awaited<ReturnType<InstanceType<typeof Apple.SignedDataVerifier>["verifyAndDecodeTransaction"]>>;
    } | null = null;
    for (const environment of [Apple.Environment.PRODUCTION, Apple.Environment.SANDBOX]) {
      try {
        const checked = new Apple.SignedDataVerifier(
          roots, false, environment, bundleID,
          environment === Apple.Environment.PRODUCTION ? appAppleID : undefined,
        );
        const transaction = await checked.verifyAndDecodeTransaction(signed);
        device = { environment, checked, transaction };
        break;
      } catch {
        // The other environment may match; otherwise the transaction is invalid.
      }
    }
    if (!device || device.transaction.productId !== productID || !device.transaction.transactionId) {
      return json({ active: false });
    }
    if (!env.APPLE_ISSUER_ID || !env.APPLE_KEY_ID || !env.APPLE_PRIVATE_KEY) {
      return json({ error: "service_unavailable" }, 503);
    }
    try {
      const client = new Apple.AppStoreServerAPIClient(
        env.APPLE_PRIVATE_KEY,
        env.APPLE_KEY_ID,
        env.APPLE_ISSUER_ID,
        bundleID,
        device.environment,
      );
      const response = await client.getTransactionInfo(device.transaction.transactionId);
      if (!response.signedTransactionInfo) return json({ error: "apple_unavailable" }, 503);
      const current = await device.checked.verifyAndDecodeTransaction(response.signedTransactionInfo);
      const active = current.transactionId === device.transaction.transactionId
        && current.originalTransactionId === device.transaction.originalTransactionId
        && current.productId === productID
        && current.revocationDate == null;
      return json({ active, environment: device.environment });
    } catch {
      // A failed live check never grants access, and Apple outages are retryable.
      return json({ error: "apple_unavailable" }, 503);
    }
  },
} satisfies ExportedHandler<Env>;
