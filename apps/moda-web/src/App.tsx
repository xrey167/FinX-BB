import { useEffect, useRef, useState } from 'react'
import type { CSSProperties, MouseEvent as ReactMouseEvent, RefObject } from 'react'

type Product = {
  id: string
  name: string
  price: number
  image: string
  alt: string
  sizes: string[]
  colors: { name: string; value: string }[]
  fit: string
}

type Curation = {
  id: string
  title: string
  kicker: string
  intro: string
  description: string
  hero: string
  heroAlt: string
  products: Product[]
}

type CartLine = {
  key: string
  product: Product
  size: string
  color: string
  quantity: number
}

const dressColors = [
  { name: 'Schwarz', value: '#171714' },
  { name: 'Elfenbein', value: '#ebe7e1' },
]
const blazerColors = [
  { name: 'Elfenbein', value: '#e8e3da' },
  { name: 'Schwarz', value: '#171714' },
  { name: 'Taupe', value: '#bbae9f' },
]
const bagColors = [
  { name: 'Schwarz', value: '#171714' },
  { name: 'Burgunder', value: '#741d37' },
  { name: 'Taupe', value: '#bbae9f' },
]

const curations: Curation[] = [
  {
    id: 'mitte-evening',
    title: 'Abend in Mitte',
    kicker: 'Berlin · 18°',
    intro: 'Ein Look, der sich nach dir anfühlt.',
    description: 'Kuratiert für Berlin, 18° und einen langen Abend.',
    hero: '/assets/moda-evening-hero.png',
    heroAlt: 'Frau in schwarzem Kleid und elfenbeinfarbenem Blazer vor Berliner Architektur',
    products: [
      { id: 'dress', name: 'Silk Column Dress', price: 420, image: '/assets/silk-column-dress.png', alt: 'Schwarzes, bodenlanges Seidenkleid mit schmalen Trägern', sizes: ['34', '36', '38', '40'], colors: dressColors, fit: 'Fällt normal aus' },
      { id: 'blazer', name: 'Sculpted Blazer', price: 590, image: '/assets/sculpted-blazer.png', alt: 'Taillierter Blazer in warmem Elfenbein', sizes: ['34', '36', '38', '40'], colors: blazerColors, fit: 'Skulpturale Passform' },
      { id: 'bag', name: 'Arc Mini Bag', price: 280, image: '/assets/arc-mini-bag.png', alt: 'Kleine schwarze Handtasche mit geschwungenem Griff', sizes: ['One Size'], colors: bagColors, fit: 'Kompaktes Format' },
    ],
  },
  {
    id: 'modern-daylight',
    title: 'Moderne Eleganz',
    kicker: 'Berlin · 19°',
    intro: 'Präzise Linien. Ganz mühelos.',
    description: 'Eine helle Silhouette für Galerie, Lunch und lange Wege.',
    hero: '/assets/sculpted-blazer.png',
    heroAlt: 'Elfenbeinfarbener Blazer als Mittelpunkt einer neuen Kuration',
    products: [
      { id: 'day-blazer', name: 'Daylight Sculpted Blazer', price: 540, image: '/assets/sculpted-blazer.png', alt: 'Taillierter Blazer in warmem Elfenbein', sizes: ['34', '36', '38', '40'], colors: blazerColors, fit: 'Skulpturale Passform' },
      { id: 'wine-bag', name: 'Burgundy Arc Bag', price: 310, image: '/assets/arc-mini-bag.png', alt: 'Kleine geschwungene Handtasche', sizes: ['One Size'], colors: bagColors, fit: 'Kompaktes Format' },
      { id: 'midi-dress', name: 'Silk Midi Dress', price: 450, image: '/assets/silk-column-dress.png', alt: 'Schwarzes Seidenkleid mit schmalen Trägern', sizes: ['34', '36', '38', '40'], colors: dressColors, fit: 'Fällt normal aus' },
    ],
  },
  {
    id: 'timeless-noir',
    title: 'Zeitlos in Schwarz',
    kicker: 'Berlin · 17°',
    intro: 'Ein Abend, auf das Wesentliche reduziert.',
    description: 'Monochrome Ruhe, getragen von Seide und klaren Konturen.',
    hero: '/assets/silk-column-dress.png',
    heroAlt: 'Schwarzes Seidenkleid als Mittelpunkt einer monochromen Kuration',
    products: [
      { id: 'noir-dress', name: 'Noir Column Dress', price: 460, image: '/assets/silk-column-dress.png', alt: 'Schwarzes, bodenlanges Seidenkleid', sizes: ['34', '36', '38', '40'], colors: dressColors, fit: 'Fällt körpernah aus' },
      { id: 'night-bag', name: 'Night Arc Mini Bag', price: 295, image: '/assets/arc-mini-bag.png', alt: 'Kleine schwarze Handtasche', sizes: ['One Size'], colors: bagColors, fit: 'Kompaktes Format' },
      { id: 'atelier-blazer', name: 'Atelier Evening Blazer', price: 620, image: '/assets/sculpted-blazer.png', alt: 'Elfenbeinfarbener Abendblazer', sizes: ['34', '36', '38', '40'], colors: blazerColors, fit: 'Gerade Schulter, schmale Taille' },
    ],
  },
]

const productCatalog = new Map(curations.flatMap((curation) => curation.products).map((product) => [product.id, product]))

const money = new Intl.NumberFormat('de-DE', {
  style: 'currency',
  currency: 'EUR',
  maximumFractionDigits: 0,
})

const focusableSelector = [
  'a[href]',
  'button:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  'input:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',')

function initialSizes(products: Product[]) {
  return Object.fromEntries(products.filter((product) => product.sizes.length === 1).map((product) => [product.id, product.sizes[0]]))
}

function initialColors(products: Product[]) {
  return Object.fromEntries(products.map((product) => [product.id, product.colors[0].name]))
}

function useModalFocus(
  active: boolean,
  containerRef: RefObject<HTMLElement | null>,
  openerRef: RefObject<HTMLElement | null>,
  onClose: () => void,
) {
  const closeRef = useRef(onClose)
  closeRef.current = onClose

  useEffect(() => {
    if (!active) return
    const container = containerRef.current
    if (!container) return
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'

    const focusables = () => Array.from(container.querySelectorAll<HTMLElement>(focusableSelector))
      .filter((element) => !element.hasAttribute('disabled') && element.getAttribute('aria-hidden') !== 'true')
    focusables()[0]?.focus()

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        closeRef.current()
        return
      }
      if (event.key !== 'Tab') return
      const items = focusables()
      if (!items.length) {
        event.preventDefault()
        container.focus()
        return
      }
      const first = items[0]
      const last = items[items.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', onKeyDown, true)
    return () => {
      document.removeEventListener('keydown', onKeyDown, true)
      document.body.style.overflow = previousOverflow
      openerRef.current?.focus()
    }
  }, [active, containerRef, openerRef])
}

function Icon({ name }: { name: 'bag' | 'heart' | 'close' | 'menu' | 'refresh' | 'arrow' | 'minus' | 'plus' }) {
  const common = { width: 24, height: 24, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, 'aria-hidden': true as const }
  if (name === 'bag') return <svg {...common}><path d="M5.2 7.5h13.6l.8 13H4.4l.8-13Z"/><path d="M8.5 8V5a3.5 3.5 0 0 1 7 0v3"/></svg>
  if (name === 'heart') return <svg {...common}><path d="M20.8 5.8c-1.8-2-5-1.7-6.8.2L12 8 10 6C8.2 4.1 5 3.8 3.2 5.8c-1.7 1.9-1.5 4.8.3 6.7L12 21l8.5-8.5c1.8-1.9 2-4.8.3-6.7Z"/></svg>
  if (name === 'close') return <svg {...common}><path d="m5 5 14 14M19 5 5 19"/></svg>
  if (name === 'menu') return <svg {...common}><path d="M3 6h18M3 12h18M3 18h18"/></svg>
  if (name === 'refresh') return <svg {...common}><path d="M19 7v5h-5M5 17v-5h5"/><path d="M7.2 8.2A6.5 6.5 0 0 1 18.5 12M5.5 12a6.5 6.5 0 0 0 11.3 3.8"/></svg>
  if (name === 'minus') return <svg {...common}><path d="M5 12h14"/></svg>
  if (name === 'plus') return <svg {...common}><path d="M5 12h14M12 5v14"/></svg>
  return <svg {...common}><path d="M5 12h14M14 6l6 6-6 6"/></svg>
}

function App() {
  const [curationIndex, setCurationIndex] = useState(0)
  const [sizes, setSizes] = useState<Record<string, string>>(() => initialSizes(curations[0].products))
  const [colors, setColors] = useState<Record<string, string>>(() => initialColors(curations[0].products))
  const [cart, setCart] = useState<CartLine[]>([])
  const [archivedOrder, setArchivedOrder] = useState<CartLine[]>([])
  const [favorites, setFavorites] = useState<string[]>([])
  const [favoritesOpen, setFavoritesOpen] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [detail, setDetail] = useState<Product | null>(null)
  const [checkoutComplete, setCheckoutComplete] = useState(false)
  const drawerRef = useRef<HTMLElement>(null)
  const detailRef = useRef<HTMLDivElement>(null)
  const drawerOpenerRef = useRef<HTMLElement>(null)
  const detailOpenerRef = useRef<HTMLElement>(null)
  const menuButtonRef = useRef<HTMLButtonElement>(null)
  const current = curations[curationIndex]
  const products = current.products
  const modalOpen = drawerOpen || detail !== null

  const cartCount = cart.reduce((sum, line) => sum + line.quantity, 0)
  const cartTotal = cart.reduce((sum, line) => sum + line.product.price * line.quantity, 0)
  const lookTotal = products.reduce((sum, product) => sum + product.price, 0)
  const favoriteProducts = favorites
    .map((productId) => productCatalog.get(productId))
    .filter((product): product is Product => product !== undefined)

  function closeDrawer() {
    setDrawerOpen(false)
    setCheckoutComplete(false)
  }

  function closeDetail() {
    setDetail(null)
  }

  useModalFocus(drawerOpen, drawerRef, drawerOpenerRef, closeDrawer)
  useModalFocus(detail !== null, detailRef, detailOpenerRef, closeDetail)

  useEffect(() => {
    if (!menuOpen) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      event.preventDefault()
      setMenuOpen(false)
      menuButtonRef.current?.focus()
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [menuOpen])

  function openDrawer(event: ReactMouseEvent<HTMLElement>) {
    drawerOpenerRef.current = event.currentTarget
    setDetail(null)
    setDrawerOpen(true)
  }

  function openDetail(product: Product, opener: HTMLElement) {
    detailOpenerRef.current = opener
    setDrawerOpen(false)
    setColors((values) => values[product.id] ? values : { ...values, [product.id]: product.colors[0].name })
    if (product.sizes.length === 1) setSizes((values) => values[product.id] ? values : { ...values, [product.id]: product.sizes[0] })
    setDetail(product)
  }

  function addToCart(product: Product) {
    const size = sizes[product.id]
    if (!size) return
    const color = colors[product.id]
    const key = `${product.id}-${size}-${color}`
    setCart((lines) => {
      const match = lines.find((line) => line.key === key)
      if (match) return lines.map((line) => line.key === key ? { ...line, quantity: line.quantity + 1 } : line)
      return [...lines, { key, product, size, color, quantity: 1 }]
    })
  }

  function changeQuantity(key: string, delta: number) {
    setCart((lines) => lines
      .map((line) => line.key === key ? { ...line, quantity: line.quantity + delta } : line)
      .filter((line) => line.quantity > 0))
  }

  function toggleFavorite(productId: string) {
    setFavorites((items) => items.includes(productId) ? items.filter((id) => id !== productId) : [...items, productId])
  }

  function showFavorites() {
    setFavoritesOpen(true)
    setMenuOpen(false)
    window.setTimeout(() => document.getElementById('favorites')?.scrollIntoView?.({ behavior: 'smooth' }), 0)
  }

  function recurate() {
    const nextIndex = (curationIndex + 1) % curations.length
    const nextProducts = curations[nextIndex].products
    setCurationIndex(nextIndex)
    setSizes(initialSizes(nextProducts))
    setColors(initialColors(nextProducts))
    setFavoritesOpen(false)
  }

  function confirmCheckout() {
    setArchivedOrder(cart)
    setCart([])
    setCheckoutComplete(true)
  }

  return (
    <div className="app-shell">
      <div data-testid="app-content" inert={modalOpen ? true : undefined} aria-hidden={modalOpen ? true : undefined}>
        <header className="site-header">
          <button ref={menuButtonRef} className="icon-button mobile-only" type="button" aria-label={menuOpen ? 'Menü schließen' : 'Menü öffnen'} aria-expanded={menuOpen} aria-controls="mobile-menu" onClick={() => setMenuOpen((open) => !open)}><Icon name={menuOpen ? 'close' : 'menu'} /></button>
          <a className="wordmark" href="#top" aria-label="MODA Startseite">MODA</a>
          <nav className="desktop-nav" aria-label="Hauptnavigation">
            <a className="active" href="#entdecken">Entdecken</a>
            <a href="#looks">Looks</a>
            <button type="button" onClick={showFavorites}>Favoriten{favorites.length ? ` (${favorites.length})` : ''}</button>
          </nav>
          <button className="cart-trigger" type="button" onClick={openDrawer} aria-label={`Warenkorb, ${cartCount} Artikel`}>
            <span className="desktop-only">Warenkorb</span><Icon name="bag" /><span className="mobile-count">{cartCount}</span>
          </button>
          {menuOpen && <nav className="mobile-menu" id="mobile-menu" aria-label="Mobile Navigation">
            <a href="#entdecken" onClick={() => setMenuOpen(false)}>Entdecken</a>
            <a href="#looks" onClick={() => setMenuOpen(false)}>Looks</a>
            <button type="button" onClick={showFavorites}>Favoriten{favorites.length ? ` (${favorites.length})` : ''}</button>
          </nav>}
        </header>

        <main id="top">
          <section className="discovery" id="entdecken" aria-labelledby="look-title">
            <div className="hero-copy">
              <span className="hairline" aria-hidden="true" />
              <p className="eyebrow">Für dich kuratiert · {String(curationIndex + 1).padStart(2, '0')}</p>
              <h1 id="look-title">{current.intro}</h1>
              <p className="intro">{current.description}</p>
              <button className="button primary" type="button" onClick={() => document.getElementById('look-products')?.scrollIntoView({ behavior: 'smooth' })}>Look ansehen <Icon name="arrow" /></button>
              <button className="button secondary" type="button" onClick={recurate}><Icon name="refresh" /> Neu kuratieren</button>
              <div className="pagination" aria-label={`Look ${curationIndex + 1} von ${curations.length}`}>
                {curations.map((look, index) => <span key={look.id} className={index === curationIndex ? 'current' : ''}>{String(index + 1).padStart(2, '0')}</span>)}
              </div>
            </div>
            <div className={`hero-image-wrap hero-${current.id}`}>
              <img data-testid="curation-hero" src={current.hero} alt={current.heroAlt} />
              <span className="image-tick left" aria-hidden="true" />
              <span className="image-tick right" aria-hidden="true" />
              <span className="vertical-caption desktop-only" aria-hidden="true">BERLIN&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; MODA</span>
            </div>

            <div className="mobile-look-heading">
              <h2>{current.title}</h2>
              <p>{current.kicker}</p>
            </div>

            <section className="product-panel" id="look-products" aria-labelledby="product-heading">
              <div className="section-heading desktop-only"><h2 id="product-heading">Dieser Look</h2><span /></div>
              <div className="product-grid">
                {products.map((product, index) => <ProductCard
                  key={product.id}
                  product={product}
                  featured={index === 0}
                  size={sizes[product.id]}
                  color={colors[product.id]}
                  favorite={favorites.includes(product.id)}
                  onSize={(size) => setSizes((values) => ({ ...values, [product.id]: size }))}
                  onColor={(color) => setColors((values) => ({ ...values, [product.id]: color }))}
                  onAdd={() => addToCart(product)}
                  onFavorite={() => toggleFavorite(product.id)}
                  onDetail={(opener) => openDetail(product, opener)}
                />)}
              </div>
              <div className="completion mobile-only" aria-label="Look vervollständigen">
                <h2>Komplettieren Sie den Look</h2>
                <div className="completion-rail">
                  {products.slice(1).map((product) => <button key={product.id} type="button" onClick={(event) => openDetail(product, event.currentTarget)} aria-label={`${product.name} im Look ansehen`}>
                    <img src={product.image} alt="" />
                    <span>{product.name}</span>
                    <strong>{money.format(product.price)}</strong>
                  </button>)}
                </div>
              </div>
            </section>
          </section>

          {favoritesOpen && <section className="favorites-view" id="favorites" role="region" aria-label="Deine Favoriten">
            <div className="section-heading"><h2>Deine Favoriten</h2><span /></div>
            {favoriteProducts.length ? <div className="favorite-grid">{favoriteProducts.map((product) => <article key={product.id}>
              <button type="button" className="favorite-image" onClick={(event) => openDetail(product, event.currentTarget)} aria-label={`${product.name} ansehen`}><img src={product.image} alt="" /></button>
              <h3>{product.name}</h3><p>{money.format(product.price)}</p>
              <button className="text-button" type="button" onClick={() => toggleFavorite(product.id)}>Entfernen</button>
            </article>)}</div> : <p className="favorites-empty">Noch keine Favoriten – markiere ein Stück mit dem Herz.</p>}
          </section>}

          <section className="more-looks" id="looks" aria-labelledby="more-title">
            <div className="section-heading"><h2 id="more-title">Weitere Looks für dich</h2><span /><button type="button" onClick={recurate}>Looks entdecken <Icon name="arrow" /></button></div>
            <div className="look-editorial-grid">
              <article><img src="/assets/sculpted-blazer.png" alt="Elfenbeinfarbener Blazer" /><div><p>Moderne<br />Eleganz</p><span /></div></article>
              <article><img src="/assets/moda-evening-hero.png" alt="Abendlook in schwarzer Seide" /><div><p>Zeitlos<br />In Schwarz</p><span /></div></article>
            </div>
          </section>
        </main>

        <footer className="look-bar">
          <p><span>3 Teile</span><i>·</i><strong>{money.format(lookTotal)}</strong></p>
          <div className="miniatures desktop-only">{products.map((product) => <img key={product.id} src={product.image} alt="" />)}</div>
          <button type="button" onClick={openDrawer}>Zur Kasse <Icon name="arrow" /></button>
        </footer>
      </div>

      {drawerOpen && <>
        <div className="scrim" aria-hidden="true" onMouseDown={closeDrawer} />
        <aside ref={drawerRef} className="cart-drawer open" role="dialog" aria-modal="true" aria-labelledby="cart-title" tabIndex={-1}>
          <div className="drawer-head">
            <div><p className="eyebrow">Deine Auswahl</p><h2 id="cart-title">Warenkorb</h2></div>
            <button className="icon-button" type="button" onClick={closeDrawer} aria-label="Warenkorb schließen"><Icon name="close" /></button>
          </div>
          {checkoutComplete ? (
            <div className="checkout-success" role="status">
              <span>✓</span><h3>Danke für deine Bestellung.</h3><p>{archivedOrder.reduce((sum, line) => sum + line.quantity, 0)} Teile sind bestätigt. Dein Berliner Abend kann beginnen.</p>
              <button className="button secondary" type="button" onClick={closeDrawer}>Weiter entdecken</button>
            </div>
          ) : cart.length === 0 ? (
            <div className="empty-cart"><p>Dein Warenkorb wartet auf den perfekten Look.</p><button className="text-button" type="button" onClick={closeDrawer}>Look entdecken <Icon name="arrow" /></button></div>
          ) : (
            <>
              <div className="cart-lines">
                {cart.map((line) => <article className="cart-line" key={line.key}>
                  <img src={line.product.image} alt="" />
                  <div><h3>{line.product.name}</h3><p>{line.color} · {line.size}</p><div className="quantity"><button type="button" aria-label={`${line.product.name} entfernen`} onClick={() => changeQuantity(line.key, -1)}><Icon name="minus" /></button><span>{line.quantity}</span><button type="button" aria-label={`${line.product.name} hinzufügen`} onClick={() => changeQuantity(line.key, 1)}><Icon name="plus" /></button></div></div>
                  <strong>{money.format(line.product.price * line.quantity)}</strong>
                </article>)}
              </div>
              <div className="drawer-total"><span>Gesamtsumme</span><strong>{money.format(cartTotal)}</strong></div>
              <p className="shipping-note">Kostenloser Versand und Rückgabe</p>
              <button className="button primary checkout-button" type="button" onClick={confirmCheckout}>Bestellung bestätigen <Icon name="arrow" /></button>
            </>
          )}
        </aside>
      </>}

      {detail && <div ref={detailRef} className="dialog-layer" role="dialog" aria-modal="true" aria-labelledby="detail-title" tabIndex={-1} onMouseDown={(event) => { if (event.target === event.currentTarget) closeDetail() }}>
          <div className="detail-dialog" data-testid="detail-panel">
            <button className="icon-button detail-close" type="button" onClick={closeDetail} aria-label="Produktdetails schließen"><Icon name="close" /></button>
            <img src={detail.image} alt={detail.alt} />
            <div className="detail-copy">
              <p className="eyebrow">Dieser Look</p><h2 id="detail-title">{detail.name}</h2><p className="detail-price">{money.format(detail.price)}</p><p>Fließende Materialien, präzise Proportionen und ein leiser Auftritt – kuratiert für deinen Abend in Berlin.</p>
              <fieldset className="detail-colors"><legend>Farbe: {colors[detail.id]}</legend><div className="swatches">{detail.colors.map((choice) => <button key={choice.name} type="button" className={choice.name === colors[detail.id] ? 'selected' : ''} style={{ '--swatch': choice.value } as CSSProperties} onClick={() => setColors((values) => ({ ...values, [detail.id]: choice.name }))} aria-label={`${choice.name} wählen`} aria-pressed={choice.name === colors[detail.id]} />)}</div></fieldset>
              <label className="detail-size">Größe<select aria-label={`Größe für ${detail.name} in Produktdetails`} value={sizes[detail.id] ?? ''} onChange={(event) => setSizes((values) => ({ ...values, [detail.id]: event.target.value }))}><option value="" disabled>Größe wählen</option>{detail.sizes.map((size) => <option key={size}>{size}</option>)}</select></label>
              <div className="detail-actions">
                <button className="button primary" type="button" disabled={!sizes[detail.id]} onClick={() => addToCart(detail)}>In den Warenkorb</button>
                <button className="button secondary" type="button" aria-pressed={favorites.includes(detail.id)} onClick={() => toggleFavorite(detail.id)}>{favorites.includes(detail.id) ? 'Aus Favoriten entfernen' : 'Als Favorit speichern'}</button>
              </div>
            </div>
          </div>
        </div>}
    </div>
  )
}

type ProductCardProps = {
  product: Product
  featured: boolean
  size?: string
  color: string
  favorite: boolean
  onSize: (size: string) => void
  onColor: (color: string) => void
  onAdd: () => void
  onFavorite: () => void
  onDetail: (opener: HTMLElement) => void
}

function ProductCard({ product, featured, size, color, favorite, onSize, onColor, onAdd, onFavorite, onDetail }: ProductCardProps) {
  return (
    <article className={`product-card ${featured ? 'featured' : ''}`} id={`product-${product.id}`} data-testid={`product-${product.id}`} data-product-id={product.id} data-price={product.price}>
      <div className="product-image-button-wrap">
        <button className="product-image-button" type="button" onClick={(event) => onDetail(event.currentTarget)} aria-label={`${product.name} ansehen`}><img src={product.image} alt={product.alt} /></button>
        <button className={`favorite-button ${favorite ? 'selected' : ''}`} type="button" onClick={onFavorite} aria-label={favorite ? `${product.name} aus Favoriten entfernen` : `${product.name} zu Favoriten hinzufügen`} aria-pressed={favorite}><Icon name="heart" /></button>
        <span className="image-tick left" aria-hidden="true" /><span className="image-tick right" aria-hidden="true" />
      </div>
      <div className="product-info">
        <h3 className="product-name"><button type="button" onClick={(event) => onDetail(event.currentTarget)}>{product.name}</button></h3>
        <p className="price">{money.format(product.price)}</p>
        <fieldset className="swatches"><legend>Farbe: {color}</legend>{product.colors.map((choice) => <button key={choice.name} type="button" className={choice.name === color ? 'selected' : ''} style={{ '--swatch': choice.value } as CSSProperties} onClick={() => onColor(choice.name)} aria-label={`${choice.name} wählen`} aria-pressed={choice.name === color} />)}</fieldset>
        <p className="fit mobile-only">{product.fit}</p>
        <fieldset className="size-picker">
          <legend>Größe</legend>
          <div className="mobile-size-buttons">{product.sizes.map((choice) => <button key={choice} type="button" className={choice === size ? 'selected' : ''} onClick={() => onSize(choice)} aria-pressed={choice === size}>{choice}</button>)}</div>
          <select className="desktop-size-select" aria-label={`Größe für ${product.name}`} value={size ?? ''} onChange={(event) => onSize(event.target.value)}><option value="" disabled>Größe wählen</option>{product.sizes.map((choice) => <option key={choice}>{choice}</option>)}</select>
        </fieldset>
      </div>
      <div className="card-actions">
        {featured && <button className="button secondary mobile-only" type="button" onClick={(event) => onDetail(event.currentTarget)}>Zum Look</button>}
        <button className="button add-button" type="button" disabled={!size} onClick={onAdd}>In den Warenkorb</button>
      </div>
    </article>
  )
}

export default App
