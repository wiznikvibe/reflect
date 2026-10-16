import type { RankedProduct } from "../api";
import { formatUsd } from "../api";

interface ProductListProps {
  products: RankedProduct[];
}

function sizeLabel(product: RankedProduct): string | null {
  if (product.width_cm == null || product.depth_cm == null || product.height_cm == null) return null;
  return `${product.width_cm} × ${product.depth_cm} × ${product.height_cm} cm`;
}

export function ProductList({ products }: ProductListProps) {
  return (
    <section className="panel product-panel" aria-labelledby="products-heading">
      <header className="panel-head">
        <p className="kicker">Catalog</p>
        <h2 id="products-heading">Ranked pieces</h2>
      </header>

      {products.length === 0 ? (
        <p className="muted">Chosen products show up here in plan order, each with why it fits the room.</p>
      ) : (
        <ol className="products">
          {products.map((product, index) => {
            const size = sizeLabel(product);
            return (
              <li key={`${product.id}-${index}`} className="product">
                <p className="rank">{String(index + 1).padStart(2, "0")}</p>
                <div>
                  <h3>{product.name || product.id}</h3>
                  <p className="product-meta">
                    {[product.category?.replaceAll("_", " "), product.material, formatUsd(product.price_usd)]
                      .filter(Boolean)
                      .join(" · ")}
                  </p>
                  {product.style_tags && product.style_tags.length > 0 ? (
                    <ul className="tags">
                      {product.style_tags.map((tag) => (
                        <li key={tag}>{tag}</li>
                      ))}
                    </ul>
                  ) : null}
                  {product.why_it_fits ? <blockquote className="why">{product.why_it_fits}</blockquote> : null}
                  {product.placement_note ? <p className="placement">{product.placement_note}</p> : null}
                  {product.role || size ? (
                    <p className="product-foot">
                      {product.role ? <span>{product.role}</span> : null}
                      {size ? <span>{size}</span> : null}
                    </p>
                  ) : null}
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
