# BetaBuddy 1.5.0 (17) · App Store and TestFlight

## What to Test

- BANANA+ Kerry (72 problems) remains free. Open the gym switcher in More and select DOME Xuhui Vanke (31 problems): non-members should see Lifetime Membership, with the original gym still available after dismissal.
- Purchase or restore the existing lifetime membership using the TestFlight sandbox. A successful unlock should open the selected gym automatically. Existing lifetime members should have access without another purchase.
- Switch between both gyms, select problems, play movement simulations and change climber builds. A language change should keep the current gym.
- Check Logbook video import, on-device analysis and saved climbs after upgrading.

Includes GymPack 1.6.0 from private data commit 3a91196. DOME covers the inner hall's left slab, roof and far wall; the island has holds but no problems yet. Some roof/far-wall problems have no simulated solution for some body presets.

## Implementation and validation

The existing verified StoreKit 2 entitlement gates additional gyms. Native navigation checks cover venue, route and wall URLs; the local resource handler rejects member-only venue assets without an entitlement. Purchasing/restoring resumes the pending venue. Refund/revocation updates remove access and return an open member gym to the free gym. Five interface languages describe the benefit.

- Core package: 10 tests passed.
- iOS unit/integration coverage: 48 tests passed across the initial run and affected-suite rerun; one test requiring a real climbing clip was skipped. The final 31-test access, StoreKit and localization run passed.
- StoreKit integration: verified purchase, entitlement restoration after reopening, restore purchases and refund revocation passed with Xcode's local StoreKit test session.
- Bundled WKWebView integration: non-member switch interception, cancellation, purchase continuation into DOME, language reload and return to BANANA+ after refund passed.
- Simulator UI: checked the locked DOME row and native membership sheet, including the purchase and restore controls.
- Signed Release archive: 1.5.0 (17), GymPack 1.6.0, all 16 manifest SHA-256 hashes verified; no StoreKit test configuration in the release app.
- Real-device StoreKit sandbox and device performance remain for TestFlight validation.

## Distribution

Uploaded through Xcode Organizer on November 20, 2026 at 14:20 China time after command-line export failed to recognize the account. Apple completed processing; build ID is `882a5fa4-7581-4fae-b670-5ae0c4ae6b1b`.

At 14:30, App Store Connect confirmed that iOS 1.5.0 (17) and the existing lifetime membership metadata update were submitted together. Submission `74d3b7ec-edfd-44a2-8d7e-2b4e300adff8` is Waiting for Review. Automatic release after approval and immediate availability to all users are enabled. The existing live release remains 1.4.0 (16) until approval.

The build is also assigned to the existing internal and external TestFlight groups. External review was submitted with automatic tester notification enabled. The public link remains https://testflight.apple.com/join/WZPw69Vx; the new build is not externally available until Beta App Review approves it.

The support-site source and generated pages now describe member gyms and local attempt analysis. Deployment remains pending because the current Wrangler login cannot access the existing Worker. App Store metadata and bundled privacy text are updated. Detailed evidence and limitations are recorded in STATUS.md.
