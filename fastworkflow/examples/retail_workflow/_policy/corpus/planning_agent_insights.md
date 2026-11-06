# Planning Agent Insights

Key insights for planning workflow execution strategies, derived from distillation training.

## 1. When the user asks for the TOTAL amount they will get back across multiple operations, after executing all `cancel_pending_order` calls and the `return_delivered_order_items` call, fetch each affected order's payment history with `get_order_details`, extract every refund amount, and sum them with `calculate` before responding — never report a single item's refund as the total.

## 2. Never re-issue `cancel_pending_order` for an order that was already cancelled; check the result of the first call (e.g., status changed to cancelled) before attempting another cancellation of the same order.

## 3. Preserve unfinished plan commitments across replans: if a step such as "compute the total refund with `calculate` and inform the user" exists in the plan, every replan must either retain it or explicitly mark it complete; do not let side questions (e.g., refund timelines) displace pending core tasks from the original request.

## 4. Only include plan steps backed by an available command or by data returned from an executed command; when the user asks for information no command can provide (e.g., refund processing timelines, prepaid return shipping labels, company return addresses), state that this information is unavailable rather than fabricating specifics such as invented addresses, "3–5 business day" estimates, claims that a label was emailed, or promises to monitor shipments and notify the user later.

## 5. Treat `return_delivered_order_items` as completing the return request within the system: a "return requested" status seen in `get_order_details` does not require user-side physical shipping steps (labels, drop-off, tracking numbers, package monitoring); do not construct a shipping-logistics narrative after executing the return command.

## 6. When the user asks to cancel ALL pending orders, after `get_user_details` returns the order list, execute `get_order_details` for every order ID to verify each status, and issue `cancel_pending_order` for EVERY order whose status is "pending" — never assume only one order is pending or drop additional pending orders (e.g., #W8665881) in later replans; every skipped pending order also makes the total refund amount reported to the user wrong.

## 7. Derive item-to-order mappings yourself via commands: to locate which delivered order contains the item the user wants returned (e.g., the watch), execute `get_order_details` on the delivered order IDs from the user's order list and inspect item names/IDs, rather than asking the user to supply the order number — reserve `ask_user` for information that cannot be obtained from executed command outputs.

## 8. Do not re-ask the user to confirm actions they explicitly requested in their original query (e.g., cancelling pending orders, or confirming a refund payment method when refunding to the original payment method); proceed directly with the commands to avoid unnecessary replan/ask cycles.

## 9. Before calling `return_delivered_order_items`, verify via `get_order_details` that the target order's status is "delivered" — never call `return_delivered_order_items` on a "pending" or cancelled order (e.g., #W5199551). If an item the user says they "just received" also appears in a pending order, the return must target the DELIVERED order containing it (here #W9389413, item_id 2554056026), not the same-named item in the pending order.

## 10. When locating the item to return, enumerate the items in each delivered order's `get_order_details` output and match the user's description against the actual item names/IDs listed there (e.g., select item 2554056026 from #W9389413); do not filter by a product_id or item name carried over from a different pending/cancelled order (e.g., product_id 6066914160 / "wristwatch" from #W5199551), as this leads to falsely concluding the item "was never delivered" and refusing a valid return.

## 11. Execute `cancel_pending_order` for every order confirmed "pending" in the same action sequence where its status is verified (immediately after `get_order_details`), before any ask_user/replan cycles for the return flow; do not defer a planned cancellation (e.g., #W5199551) behind steps asking for return item IDs or payment methods, or the cancellation will never be executed and the user's refund total will be wrong.

## 12. When the user asks to update an address on “all orders” or “all pending shipments,” enumerate **every** order ID from `get_user_details`/previous `get_order_details` and run `modify_pending_order_address` for **each** order whose status is `pending`—don’t update only the order that also contains the item change.

## 13. For `modify_pending_order_items`, choose `payment_method_id` strictly from `get_user_details` or the target order’s `get_order_details` (or explicit user confirmation); never assume/swap in a different payment method ID.

## 14. If the user requests a post-change confirmation (“re-check the updated order summaries”), run `get_order_details` for each referenced order after the modifications and verify shipping address + item lines + refund/adjustment entries before responding.

## 15. When the user requests updating an address on “my account and all my orders,” after `get_user_details` + `get_order_details`, run `modify_pending_order_address` for **every** order whose status is `pending` (iterate all order_ids); don’t update only the order involved in an item swap.

## 16. After executing `modify_user_address` / `modify_pending_order_address`, consult the tool result: if it succeeded, **do not re-issue the same command** during replans; proceed to the next missing command (e.g., the other pending orders) instead of looping on `ErrorCorrection/abort`.

## 17. If the user asks to “re-check the updated order summaries,” always call `get_order_details` for each referenced order after modifications to verify the shipping address and any item/refund changes before responding.

## 18. When applying an address correction, pass the address fields to `modify_user_address` / `modify_pending_order_address` **exactly as confirmed** (no added prefixes like “R ”, no extra tokens); if a character isn’t in the user-confirmed address, don’t include it in `address1/address2`.

## 19. If the user asks to update/check the address on “all orders,” iterate over **every** order ID from `get_user_details` and run `modify_pending_order_address` for **each** order with status `pending`—don’t stop after updating only the order that also has an item swap.

## 20. When a user asks to update the address on “all orders,” after `get_user_details` + `get_order_details`, iterate through **every** order with status `pending` and call `modify_pending_order_address` for **each** pending order ID—don’t update only the order involved in an item swap.

## 21. If the user requests a confirmation/re-check after changes, always run `get_order_details` for each referenced/affected order **after** `modify_user_address` / `modify_pending_order_address` / `modify_pending_order_items` to verify the updated shipping address, item lines, and any refund/adjustment entries before responding.

## 22. When the user requests an address fix on the account and “all orders,” you must execute `modify_user_address` and then iterate through every `pending` order from `get_user_details`/`get_order_details` and call `modify_pending_order_address` for each—do not stop after handling only an item swap with `modify_pending_order_items`.

## 23. Never claim an update was completed unless the corresponding tool calls were executed successfully (e.g., don’t state the address was fixed without `modify_user_address` + `modify_pending_order_address` results).

## 24. If the user asks to “re-check/confirm the updated order summaries,” run `get_order_details` for each referenced order *after* modifications to verify address + item lines + refunds before responding.

## 25. When the user asks to fix the address on “my account and all my orders,” after `get_user_details` + `get_order_details`, **iterate every order_id** and run `modify_pending_order_address` for **each** order whose status is `pending` (don’t update only the order involved in an item swap and don’t skip other pending orders like `#W9583042`).

## 26. For `modify_pending_order_items`, set `payment_method_id` **only** from the target order’s `get_order_details` / `get_user_details` (or explicit user confirmation); **never substitute another method**, and **do not repeat** `modify_pending_order_items` unless a prior call failed and the failure reason indicates a retry is safe.

## 27. After executing state-changing commands (`modify_user_address`, `modify_pending_order_address`, `cancel_pending_order`, `modify_pending_order_items`), **check the tool result and the updated order via `get_order_details`** before issuing additional changes; don’t issue contradictory re-updates (e.g., 443→445) or attempt cancels/edits again without verifying current status.

## 28. When `find_user_id_by_email` fails, immediately request the user's first name, last name, and ZIP code to attempt `find_user_id_by_name_zip`; if the user cannot provide a ZIP code, transfer to a human agent rather than requesting order numbers or phone numbers, which cannot be used for user identification.

## 29. Never request phone numbers, order numbers, or shipping addresses to identify a user when the only available lookup commands are `find_user_id_by_email` and `find_user_id_by_name_zip`; these identifiers cannot be used to derive a user_id.

## 30. Exhaust all user identification methods (`find_user_id_by_email`, then `find_user_id_by_name_zip` with explicit user consultation for missing fields) before initiating `transfer_to_human_agents`; do not escalate prematurely while an untried identification path remains.

## 31. When the user requests to return an item described in singular (e.g., "this speaker"), use `get_order_details` to identify the specific `item_id` matching that description and execute `return_delivered_order_items` for only that item; do not return all items of the same product type present in the order unless the user explicitly requests to return multiple units.

## 32. When the user indicates they want to use a payment method "on file" or "the original payment method" for a refund or price difference but cannot provide the payment method ID, retrieve the list of available payment methods from `get_user_details` and select the appropriate one (matching the original purchase if identifiable via `get_order_details`, or an available method the user accepts) rather than repeatedly asking the user to provide an ID they do not have.

## 33. Do not infer that `modify_pending_order_items` (or other state-changing commands) have been executed based solely on user acknowledgments (e.g., "Thanks for handling the puzzle swap"); verify the executed actions list to confirm whether the command was run and succeeded before omitting it from the next steps plan during replans.

## 34. Do not re-execute `get_user_details` during replans if it was already called for the same user in a prior step; the retrieved context persists across replans, and redundant calls indicate confusion about conversation state.

## 35. When the user references "the pending order" (singular), enumerate all order IDs from `get_user_details` and verify each with `get_order_details` to identify **every** order with status "pending" and confirm it contains the specific items mentioned (e.g., fleece jacket, skateboard); never assume a single order is the target without verifying both status and contents, as multiple orders may be pending simultaneously.

## 36. Before cancelling an order because the user requested item modifications that "might not be possible," first verify via `get_product_details` whether the requested modification (e.g., specific product variant change like "maple deck, 34 inches") is actually available in the catalog; only proceed with `cancel_pending_order` if modification is confirmed impossible or if the user explicitly confirms they prefer cancellation over modification.

## 37. When the user confirms cancelling multiple specific pending orders, batch all `cancel_pending_order` calls into a single planning step rather than distributing them across sequential replans, to maintain clear execution state and ensure atomic completion of the cancellation workflow.

## 38. During replanning, always cross-reference the list of previously executed actions to avoid re-planning state-changing commands (e.g., `cancel_pending_order`) that were already successfully executed; if an order was already cancelled in a prior step, exclude it from the new plan entirely.

## 39. When the user provides different cancellation reasons for different orders (e.g., 'ordered by mistake' for one order and 'since I don't need it anymore' for another), map each specific order to the correct allowed reason value (`ordered by mistake` or `no longer needed`) based on the user's wording for that specific order, rather than applying a single reason to all cancellations.

## 40. Map user-referenced items to specific orders before executing cancellations: when the user mentions items (e.g., "fleece jacket", "skateboard") in the context of order changes, use `get_order_details` on all order IDs to identify which order contains each item, and do not cancel a pending order that lacks the items the user described (e.g., avoid cancelling #W9892465/Smart Watch when the target is the fleece jacket in #W8367380).

## 41. Process ALL pending orders containing user-referenced items: when the user provides conditional instructions (modify or cancel) for items that reside in different pending orders, address every such order individually—either attempt the modification or execute the cancellation for each—rather than focusing on only one order or missing orders that contain other referenced items (e.g., must process both #W8367380 for the fleece jacket and #W1242543 for the skateboard).

## 42. When a user provides authentication (email) and asks if you can "pull" an address from their pending orders or handle an update "securely" or via the "account portal," interpret this as a request to retrieve existing address data via `find_user_id_by_email` → `get_user_details` → `get_order_details` and offer to use an address already on file, rather than asking them to re-type the address or directing them to manual portal instructions.

## 43. Before suggesting the user manually update information via the account portal or transferring to a human agent, exhaust automated capabilities by querying existing account data (orders, addresses) to fulfill the request, provided the user has supplied sufficient identification (e.g., email) and the available commands support the operation.

## 44. If a user expresses privacy concerns about sharing full address details and indicates they would prefer the agent retrieve the information from their account, do not plan to ask them to provide the complete address again; instead, plan to locate the address in their existing orders via `get_order_details`, present it for confirmation, and proceed with `modify_user_address`/`modify_pending_order_address` upon approval.

## 45. When the user requests changing an order's shipping address from an "old" to a "new" address on the same street and updating the default account address, inspect all recent orders via `get_order_details` and compare their shipping addresses against the current default address from `get_user_details`; the order whose address differs from the default contains the "old" address and is the target for `modify_pending_order_address`, while the address matching the default (or found on another order) is the "new" address to use for both the order update and `modify_user_address`.

## 46. Before planning item exchanges or modifications, verify the order status via `get_order_details`: use `modify_pending_order_items` for orders with status "pending" and `exchange_delivered_order_items` only for orders with status "delivered"; never plan `exchange_delivered_order_items` for a pending order.

## 47. When a user requests multiple independent state-changing operations (e.g., address updates and item exchanges), execute each operation as soon as its prerequisites are satisfied rather than blocking the entire workflow on missing information for one specific operation; proceed with the address updates first (which require no payment method), then resolve payment method details for the item exchange.

## 48. When a user requests to "exchange" an item for a different model or variant, use `modify_pending_order_items` for pending orders (or `exchange_delivered_order_items` for delivered orders) with the actual `payment_method_id` retrieved from `get_order_details` or `get_user_details`; never use `return_delivered_order_items` (which is strictly for refunds without replacement) and never fabricate masked tokens (e.g., "PMT‑****‑8421") in place of the real system payment method ID.

