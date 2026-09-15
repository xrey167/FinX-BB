# MODA Web MVP

Responsive React/Vite storefront prototype based on the MODA desktop and mobile design concepts.

Use Node.js 24.15 or newer on the 24.x line (Node.js 22.22.2+ on the 22.x line, or 26+,
is supported as well). This matches the checked-in Vite/jsdom toolchain.

```bash
npm ci
npm run dev
```

Verification:

```bash
npm test
npm run build
```

The checkout is intentionally simulated in local UI state; no payment or backend request is made.

## Design asset note

The mobile concept shows a shoe in its completion rail, but the supplied production assets contain only the dress, blazer and bag. The featured product is already fully configurable directly above the rail, so the rail deliberately contains the two complementary real products instead of duplicating the featured item or inventing unsupported footwear. Both rail products open fully configurable product details and can be purchased there.

## Browser evidence

Responsive CSS and interactions are structurally verified by the component test suite and production build. A real-browser screenshot-regression pass at the desktop reference size and at 390 × 844 remains follow-up work: the available cloud browser could not reach the local preview URL, and the Chromium download CDN was unavailable. No visual browser-pass is claimed by this MVP.
