import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import App from './App'
import styles from './styles.css?raw'

function productCard(name: string) {
  const heading = screen.getByRole('heading', { name })
  const card = heading.closest('article')
  if (!card) throw new Error(`Produktkarte für ${name} fehlt`)
  return within(card)
}

describe('MODA commerce flow', () => {
  it('declares explicit two-column action spanning for the tablet source breakpoint', () => {
    const tabletBlock = styles.match(/@media \(max-width: 1100px\) and \(min-width: 721px\) \{([\s\S]*?)\n\}/)?.[1]
    expect(tabletBlock).toMatch(/\.product-card > \.card-actions\s*\{[^}]*grid-column:\s*1\s*\/\s*-1/)
  })

  it('gates sized products until a size is selected', async () => {
    const user = userEvent.setup()
    render(<App />)
    const dress = productCard('Silk Column Dress')
    const add = dress.getByRole('button', { name: 'In den Warenkorb' })

    expect(add).toBeDisabled()
    await user.selectOptions(dress.getByRole('combobox', { name: 'Größe für Silk Column Dress' }), '38')
    expect(add).toBeEnabled()
  })

  it('calculates the cart total across the full curated look', async () => {
    const user = userEvent.setup()
    render(<App />)

    const dress = productCard('Silk Column Dress')
    await user.selectOptions(dress.getByRole('combobox'), '38')
    await user.click(dress.getByRole('button', { name: 'In den Warenkorb' }))

    const blazer = productCard('Sculpted Blazer')
    await user.selectOptions(blazer.getByRole('combobox'), '38')
    await user.click(blazer.getByRole('button', { name: 'In den Warenkorb' }))

    await user.click(productCard('Arc Mini Bag').getByRole('button', { name: 'In den Warenkorb' }))
    await user.click(screen.getByRole('button', { name: /Warenkorb, 3 Artikel/ }))

    const drawer = screen.getByRole('dialog', { name: 'Warenkorb' })
    expect(within(drawer).getByText('Gesamtsumme')).toBeInTheDocument()
    expect(within(drawer).getByText(/1\.290/)).toBeInTheDocument()
  })

  it('recurates hero and real product data, then wraps to the first curation', async () => {
    const user = userEvent.setup()
    render(<App />)
    const hero = screen.getByTestId('curation-hero')

    expect(screen.getByText('Abend in Mitte')).toBeInTheDocument()
    expect(hero).toHaveAttribute('src', '/assets/moda-evening-hero.png')
    expect(screen.getByTestId('product-dress')).toHaveAttribute('data-price', '420')

    await user.click(screen.getByRole('button', { name: /Neu kuratieren/ }))
    expect(screen.getByText('Moderne Eleganz')).toBeInTheDocument()
    expect(hero).toHaveAttribute('src', '/assets/sculpted-blazer.png')
    expect(screen.queryByTestId('product-dress')).not.toBeInTheDocument()
    expect(screen.getByTestId('product-day-blazer')).toHaveAttribute('data-price', '540')

    await user.click(screen.getByRole('button', { name: /Neu kuratieren/ }))
    expect(screen.getByText('Zeitlos in Schwarz')).toBeInTheDocument()
    expect(hero).toHaveAttribute('src', '/assets/silk-column-dress.png')
    expect(screen.getByTestId('product-noir-dress')).toHaveAttribute('data-price', '460')

    await user.click(screen.getByRole('button', { name: /Neu kuratieren/ }))
    expect(screen.getByText('Abend in Mitte')).toBeInTheDocument()
    expect(screen.getByTestId('product-dress')).toHaveAttribute('data-price', '420')
    expect(screen.getByLabelText('Look 1 von 3')).toBeInTheDocument()
  })

  it('traps focus in product details, closes on Escape, unlocks body and restores focus', async () => {
    const user = userEvent.setup()
    render(<App />)
    const opener = screen.getByRole('button', { name: 'Silk Column Dress ansehen' })

    await user.click(opener)
    const dialog = screen.getByRole('dialog', { name: 'Silk Column Dress' })
    const close = within(dialog).getByRole('button', { name: 'Produktdetails schließen' })
    const last = within(dialog).getByRole('button', { name: /Als Favorit speichern/ })
    expect(dialog).toHaveAttribute('aria-modal', 'true')
    expect(screen.getByTestId('app-content')).toHaveAttribute('inert')
    expect(document.body.style.overflow).toBe('hidden')
    expect(close).toHaveFocus()

    await user.tab({ shift: true })
    expect(last).toHaveFocus()
    await user.tab()
    expect(close).toHaveFocus()
    await user.keyboard('{Escape}')

    expect(screen.queryByRole('dialog', { name: 'Silk Column Dress' })).not.toBeInTheDocument()
    expect(document.body.style.overflow).toBe('')
    expect(opener).toHaveFocus()
  })

  it('gives the cart drawer modal focus semantics and restores its opener', async () => {
    const user = userEvent.setup()
    render(<App />)
    const opener = screen.getByRole('button', { name: /Warenkorb, 0 Artikel/ })

    await user.click(opener)
    const dialog = screen.getByRole('dialog', { name: 'Warenkorb' })
    expect(dialog).toHaveAttribute('aria-modal', 'true')
    expect(within(dialog).getByRole('button', { name: 'Warenkorb schließen' })).toHaveFocus()
    await user.keyboard('{Escape}')
    expect(screen.queryByRole('dialog', { name: 'Warenkorb' })).not.toBeInTheDocument()
    expect(opener).toHaveFocus()
  })

  it('opens a working menu and shows a genuinely filtered favorites section', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(productCard('Silk Column Dress').getByRole('button', { name: /zu Favoriten hinzufügen/ }))
    const menu = screen.getByRole('button', { name: 'Menü öffnen' })
    await user.click(menu)
    expect(menu).toHaveAttribute('aria-expanded', 'true')
    const mobileNav = screen.getByRole('navigation', { name: 'Mobile Navigation' })
    await user.click(within(mobileNav).getByRole('button', { name: /Favoriten/ }))

    const favorites = screen.getByRole('region', { name: 'Deine Favoriten' })
    expect(within(favorites).getByRole('heading', { name: 'Silk Column Dress' })).toBeInTheDocument()
    expect(within(favorites).queryByRole('heading', { name: 'Sculpted Blazer' })).not.toBeInTheDocument()
  })

  it('keeps favorites resolvable from the full catalog after recuration', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(productCard('Silk Column Dress').getByRole('button', { name: /zu Favoriten hinzufügen/ }))
    await user.click(screen.getByRole('button', { name: /Neu kuratieren/ }))
    await user.click(screen.getByRole('button', { name: 'Favoriten (1)' }))

    const favorites = screen.getByRole('region', { name: 'Deine Favoriten' })
    expect(within(favorites).getByRole('heading', { name: 'Silk Column Dress' })).toBeInTheDocument()
    expect(within(favorites).getAllByRole('article')).toHaveLength(1)
  })

  it('configures and buys complementary blazer and bag products through rail details', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: 'Sculpted Blazer im Look ansehen' }))
    let dialog = screen.getByRole('dialog', { name: 'Sculpted Blazer' })
    const blazerAdd = within(dialog).getByRole('button', { name: 'In den Warenkorb' })
    expect(blazerAdd).toBeDisabled()
    await user.selectOptions(within(dialog).getByRole('combobox', { name: 'Größe für Sculpted Blazer in Produktdetails' }), '38')
    await user.click(within(dialog).getByRole('button', { name: 'Schwarz wählen' }))
    await user.click(blazerAdd)
    await user.click(within(dialog).getByRole('button', { name: 'Produktdetails schließen' }))

    await user.click(screen.getByRole('button', { name: 'Arc Mini Bag im Look ansehen' }))
    dialog = screen.getByRole('dialog', { name: 'Arc Mini Bag' })
    await user.click(within(dialog).getByRole('button', { name: 'In den Warenkorb' }))
    await user.click(within(dialog).getByRole('button', { name: 'Produktdetails schließen' }))

    expect(screen.getByLabelText('Warenkorb, 2 Artikel')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /Warenkorb, 2 Artikel/ }))
    expect(within(screen.getByRole('dialog', { name: 'Warenkorb' })).getByText(/870/)).toBeInTheDocument()
  })

  it('persists a rail bag color and favorite selection into global commerce state', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: 'Arc Mini Bag im Look ansehen' }))
    const dialog = screen.getByRole('dialog', { name: 'Arc Mini Bag' })
    await user.click(within(dialog).getByRole('button', { name: 'Burgunder wählen' }))
    const favorite = within(dialog).getByRole('button', { name: 'Als Favorit speichern' })
    await user.click(favorite)
    expect(favorite).toHaveAttribute('aria-pressed', 'true')
    await user.click(within(dialog).getByRole('button', { name: 'In den Warenkorb' }))
    await user.click(within(dialog).getByRole('button', { name: 'Produktdetails schließen' }))

    expect(screen.getByRole('button', { name: 'Favoriten (1)' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /Warenkorb, 1 Artikel/ }))
    expect(within(screen.getByRole('dialog', { name: 'Warenkorb' })).getByText('Burgunder · One Size')).toBeInTheDocument()
  })

  it('closes product details only from the full-screen backdrop, not dialog content', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('button', { name: 'Silk Column Dress ansehen' }))
    const dialog = screen.getByRole('dialog', { name: 'Silk Column Dress' })
    const panel = within(dialog).getByTestId('detail-panel')

    fireEvent.mouseDown(panel)
    expect(screen.getByRole('dialog', { name: 'Silk Column Dress' })).toBeInTheDocument()
    fireEvent.mouseDown(dialog)
    expect(screen.queryByRole('dialog', { name: 'Silk Column Dress' })).not.toBeInTheDocument()
  })

  it('closes the mobile menu on Escape and restores focus to its trigger', async () => {
    const user = userEvent.setup()
    render(<App />)
    const menu = screen.getByRole('button', { name: 'Menü öffnen' })

    await user.click(menu)
    expect(screen.getByRole('navigation', { name: 'Mobile Navigation' })).toBeInTheDocument()
    await user.keyboard('{Escape}')

    expect(screen.queryByRole('navigation', { name: 'Mobile Navigation' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Menü öffnen' })).toHaveFocus()
  })

  it('closes only the topmost drawer before the underlying mobile menu', async () => {
    const user = userEvent.setup()
    render(<App />)
    const menu = screen.getByRole('button', { name: 'Menü öffnen' })
    await user.click(menu)
    const cart = screen.getByRole('button', { name: /Warenkorb, 0 Artikel/ })
    await user.click(cart)

    expect(screen.getByRole('dialog', { name: 'Warenkorb' })).toBeInTheDocument()
    await user.keyboard('{Escape}')
    expect(screen.queryByRole('dialog', { name: 'Warenkorb' })).not.toBeInTheDocument()
    expect(screen.getByRole('navigation', { name: 'Mobile Navigation' })).toBeInTheDocument()
    expect(cart).toHaveFocus()

    await user.keyboard('{Escape}')
    expect(screen.queryByRole('navigation', { name: 'Mobile Navigation' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Menü öffnen' })).toHaveFocus()
  })

  it('archives and clears the cart when checkout is confirmed', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(productCard('Arc Mini Bag').getByRole('button', { name: 'In den Warenkorb' }))
    await user.click(screen.getByRole('button', { name: /Warenkorb, 1 Artikel/ }))
    await user.click(screen.getByRole('button', { name: /Bestellung bestätigen/ }))

    expect(screen.getByRole('status')).toHaveTextContent('Danke für deine Bestellung.')
    expect(screen.getByLabelText('Warenkorb, 0 Artikel')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Weiter entdecken' }))
    await user.click(screen.getByRole('button', { name: /Warenkorb, 0 Artikel/ }))
    expect(screen.getByText('Dein Warenkorb wartet auf den perfekten Look.')).toBeInTheDocument()
  })
})
