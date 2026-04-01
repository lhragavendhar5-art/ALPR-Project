/**
 * ALPR Dashboard — JavaScript API Reference
 * All API calls made by the dashboard to main.py (Flask backend)
 *
 * Base URL: http://localhost:5000
 *
 * ══════════════════════════════════════════════════════════
 *  HEALTH & STATUS
 * ══════════════════════════════════════════════════════════
 * GET  /api/health             → { status, timestamp, total_vehicles, total_events, camera_active, gate }
 * GET  /api/statistics         → { total_registered, total_events, today_entries, today_exits, today_unknown, gate }
 * GET  /api/engine-status      → { yolo_enabled, yolo_model, frcnn_enabled, cnn_enabled, pipeline, ... }
 * GET  /api/cnn-status         → { cnn_enabled, cnn_chars, pipeline }
 *
 * ══════════════════════════════════════════════════════════
 *  SCANNING
 * ══════════════════════════════════════════════════════════
 * POST /api/scan-image         → { success, plate, confidence, vehicle, event, gate, notifications, image, model_meta }
 *   body: { image_data: "<base64>", event_type: "ENTRY"|"EXIT" }
 *   body (multipart): image file + event_type field
 *
 * POST /api/manual-entry       → { success, plate, vehicle, event, gate, notifications, source:"manual_override" }
 *   body: { plate: "TN09AB1234", event_type: "ENTRY"|"EXIT", operator: "Dashboard Operator" }
 *
 * GET  /api/fuzzy-search?q=TN09  → { matches: [{ plate, owner, flat, score }], query }
 *
 * ══════════════════════════════════════════════════════════
 *  CAMERA
 * ══════════════════════════════════════════════════════════
 * GET  /api/camera/list        → { cameras: [{ id, resolution, label }], count }
 * POST /api/camera/start       → { success, message }
 *   body: { camera_id: 0 }
 * POST /api/camera/stop        → { success }
 * GET  /api/camera/status      → { running, has_frame }
 * GET  /api/camera/feed        → multipart/x-mixed-replace MJPEG stream
 * POST /api/camera/scan        → same as scan-image response
 *   body: { event_type: "ENTRY"|"EXIT" }
 *
 * ══════════════════════════════════════════════════════════
 *  VEHICLES
 * ══════════════════════════════════════════════════════════
 * GET  /api/all-vehicles       → { total, vehicles: [...] }
 * GET  /api/vehicle/<plate>    → { plate, owner_name, flat_number, vehicle_type, color, owner_phone, ... }
 * POST /api/register-vehicle   → { success, message, vehicle }
 *   body: { plate_number, owner_name, owner_phone, owner_email, vehicle_type, color, flat_number,
 *           notify_channels, telegram_chat_id, whatsapp_number }
 * PUT  /api/vehicle/<plate>    → { success, vehicle }
 * DELETE /api/vehicle/<plate>  → { success }
 *
 * ══════════════════════════════════════════════════════════
 *  EVENTS & NOTIFICATIONS
 * ══════════════════════════════════════════════════════════
 * GET  /api/events?limit=50    → { total, events: [...] }
 * POST /api/log-event          → { success, event, gate, notifications }
 *   body: { plate, event_type, confidence }
 * GET  /api/notifications      → { notifications: [...], unread_count }
 * POST /api/notifications/mark-read → { success }
 *
 * ══════════════════════════════════════════════════════════
 *  GATE
 * ══════════════════════════════════════════════════════════
 * GET  /api/gate/status        → { state, last_plate, last_action, mode }
 * POST /api/gate/open          → { success, message, gate }
 *   body: { plate: "MANUAL" }
 * GET  /api/gate/config        → { gate_mode, gate_webhook, gate_open_ms }
 * POST /api/gate/config        → { success }
 *   body: { gate_mode, gate_webhook, gate_open_ms }
 *
 * ══════════════════════════════════════════════════════════
 *  MODEL COMPARISON
 * ══════════════════════════════════════════════════════════
 * GET  /api/model-comparison   → { yolo_loaded, frcnn_loaded, history, summary }
 */
