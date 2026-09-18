const BASE = "http://localhost:8000";

export async function predictDelay(flight) { //async is basically a promise (ocaml functional programming)
    const res = await fetch(`${BASE}/predict`, { //await = pause here until the server responds, then continue
        method: "POST", // just like @app.post in FastAPI
        headers: {"Constant-Type": "application/json"}, //telling server "im sending json"
        body: JSON.stringify(flight) // turning JS object into string for wire
    });
    if (!res.ok) throw new Error(await res.text()); // error handling HTTR
    return res.json(); // turning the server's response back into JS object 
    
}

// () => is shorthand to write function that takes no inputs
export const fetchAirlines = () => fetch(`${BASE}/analytics/airlines`).then(r => r.json());
export const fetchHourly   = () => fetch(`${BASE}/analytics/hourly`).then(r => r.json());
export const fetchHeatmap  = () => fetch(`${BASE}/analytics/heatmap`).then(r => r.json());
export const fetchRoutes   = (n = 20) => fetch(`${BASE}/analytics/routes?top_n=${n}`).then(r => r.json());