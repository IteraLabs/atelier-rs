//! Benchmark for Orderbook Methods

use atelier_data::orderbooks::Orderbook;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn create_orderbook(c: &mut Criterion) {
    let mut group = c.benchmark_group("Orderbook Creation");

    let v_bids_levels = [Some((2, 5)), Some((2, 5)), Some((5, 10)), Some((10, 20))];

    let v_asks_levels = [Some((2, 5)), Some((2, 5)), Some((5, 10)), Some((10, 20))];

    let v_bids_orders = [
        Some((5, 10)),
        Some((100, 200)),
        Some((500, 700)),
        Some((1000, 1300)),
    ];
    let v_asks_orders = [
        Some((5, 10)),
        Some((100, 200)),
        Some((500, 700)),
        Some((1000, 1300)),
    ];

    let ref_bid_price = 100_000.00;
    let ref_ask_price = 100_000.10;

    // Benchmark with different parameters
    for bids_orders in v_bids_orders.iter() {
        for asks_orders in v_asks_orders.iter() {
            for bids_levels in v_bids_levels.iter() {
                for asks_levels in v_asks_levels.iter() {
                    let id = format!(
                        "b_l_{:?}_b_o_{:?}_a_l_{:?}_a_o_{:?}",
                        bids_levels, bids_orders, asks_levels, asks_orders
                    );
                    group.bench_with_input(
                        criterion::BenchmarkId::new("random", id),
                        &(*bids_levels, *bids_orders, *asks_levels, *asks_orders),
                        |b, &(bids_l, bids_o, asks_l, asks_o)| {
                            b.iter(|| {
                                Orderbook::random(
                                    Some(0),                  // update_ts
                                    black_box(ref_bid_price), // bids_price
                                    black_box(bids_l),        // bids_levels
                                    black_box(bids_o),        // bids_orders
                                    black_box(None),          // tick_size
                                    black_box(ref_ask_price), // asks_price
                                    black_box(asks_l),        // asks_levels
                                    black_box(asks_o),        // asks_orders
                                )
                            });
                        },
                    );
                }
            }
        }
    }
    group.finish();
}

criterion_group!(benches, create_orderbook);
criterion_main!(benches);
