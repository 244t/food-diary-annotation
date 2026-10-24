# Annotation guidelines

These are the dish-counting and naming conventions used in *Automating Dish-Level Annotation of Real Food-Diary Photos*.

## Dish counting and boxes

| Case | Counting rule |
| --- | --- |
| Dishes in separate containers | Count separately. |
| A main dish and vegetable garnish on the same plate | Count separately, including small sides such as shredded cabbage and pickles. |
| A main dish served on salad greens | Count the main dish and salad separately. |
| Meat and vegetables cooked together, such as a stir-fry | Count as one dish. |
| Rice bowls, curry rice, fried rice, and omurice | Count as one dish; keep rice and topping/sauce together. |
| Soup with mixed ingredients | Count as one dish. |
| Several pieces of the same food, or an assortment of one kind of food | Count as one dish. |

Record a separate axis-aligned bounding box for each dish under these rules.
The single/multi-dish classifier labels a crop with one dish as single-dish and a crop with two or more dishes as multi-dish. It concerns the main container at the center; neighboring foods cut off by the crop edges are ignored for this routing decision.

Record boxes as `[x1, y1, x2, y2]` in pixels, measured from the top-left of the annotated crop or photo.

## Dish names

Assign the exact taxonomy name when a suitable entry exists. Otherwise, record no match (`null`). Keep parenthetical variants as distinct names.
