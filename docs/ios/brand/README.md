# BetaBuddy mark

A heavy, forward-leaning open climbing link. The diagonal opening gives the silhouette a deliberate break; a single solid shape works at both navigation and home-screen sizes. No lettering, decorative arrow, gradient, or external font is required.

The canonical vector is `ios/BetaBuddy/Views/BrandMark.swift`. The app header and PNG generator use the same geometry. The app icon is opaque sRGB, with the ink mark on the brand lime field; iOS applies its own corner mask.

Regenerate from the repository root:

```sh
swiftc ios/BetaBuddy/Views/BrandMark.swift ios/scripts/generate-icon.swift -o /tmp/betabuddy-icon
/tmp/betabuddy-icon ios/BetaBuddy/Resources/Assets.xcassets/AppIcon.appiconset/AppIcon.png
```
