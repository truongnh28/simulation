import React, { useState, useEffect, useRef } from 'react';

// --- ICONS ---
const IconPlay = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>;
const IconPause = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>;
const IconStepForward = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 4 15 12 5 20 5 4"></polygon><line x1="19" y1="5" x2="19" y2="19"></line></svg>;
const IconStepBack = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="19 20 9 12 19 4 19 20"></polygon><line x1="5" y1="19" x2="5" y2="5"></line></svg>;
const IconReset = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path><polyline points="3 3 3 8 8 8"></polyline></svg>;
const IconSettings = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>;
const IconPlayCircle = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><polygon points="10 8 16 12 10 16 10 8"></polygon></svg>;

// --- DEFAULT DATA ---
const DEFAULT_STATES = ['Khỏe', 'Sốt'];
const DEFAULT_OBS = ['Bình thường', 'Lạnh', 'Chóng mặt'];
const DEFAULT_PI = [0.6, 0.4];
const DEFAULT_A = [[0.7, 0.3], [0.4, 0.6]];
const DEFAULT_B = [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]];
const DEFAULT_SEQ = ['Bình thường', 'Lạnh', 'Chóng mặt'];

export default function App() {
    const [view, setView] = useState('config'); 
    
    const [statesStr, setStatesStr] = useState(DEFAULT_STATES.join(', '));
    const [obsStr, setObsStr] = useState(DEFAULT_OBS.join(', '));
    const [seqStr, setSeqStr] = useState(DEFAULT_SEQ.join(', '));
    
    const [states, setStates] = useState(DEFAULT_STATES);
    const [observations, setObservations] = useState(DEFAULT_OBS);
    const [pi, setPi] = useState(DEFAULT_PI);
    const [A, setA] = useState(DEFAULT_A);
    const [B, setB] = useState(DEFAULT_B);
    const [sequence, setSequence] = useState(DEFAULT_SEQ);

    const [errorMsg, setErrorMsg] = useState("");

    const [simSteps, setSimSteps] = useState([]);
    const [currentStepIdx, setCurrentStepIdx] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const timerRef = useRef(null);

    const handleUpdateLists = () => {
        const newStates = statesStr.split(',').map(s => s.trim()).filter(s => s);
        const newObs = obsStr.split(',').map(s => s.trim()).filter(s => s);
        
        if (newStates.length === 0 || newObs.length === 0) {
            setErrorMsg("Danh sách trạng thái và quan sát không được để trống!");
            return;
        }

        let newPi = [...pi];
        while(newPi.length < newStates.length) newPi.push(0);
        newPi = newPi.slice(0, newStates.length);

        let newA = A.map(row => {
            let newRow = [...row];
            while(newRow.length < newStates.length) newRow.push(0);
            return newRow.slice(0, newStates.length);
        });
        while(newA.length < newStates.length) newA.push(new Array(newStates.length).fill(0));
        newA = newA.slice(0, newStates.length);

        let newB = B.map(row => {
            let newRow = [...row];
            while(newRow.length < newObs.length) newRow.push(0);
            return newRow.slice(0, newObs.length);
        });
        while(newB.length < newStates.length) newB.push(new Array(newObs.length).fill(0));
        newB = newB.slice(0, newStates.length);

        setStates(newStates);
        setObservations(newObs);
        setPi(newPi);
        setA(newA);
        setB(newB);
        setErrorMsg("");
    };

    const formatProb = (p) => {
        if (p === null || p === undefined) return "";
        if (p === 0 || p === "0") return "0";
        let num = parseFloat(p);
        if (isNaN(num)) return "0";
        if (num === 0) return "0";
        if (num < 0.0001) return num.toExponential(3);
        return num.toFixed(4).replace(/\.?0+$/, "");
    };

    const generateSimulation = () => {
        const parsedSeq = seqStr.split(',').map(s => s.trim()).filter(s => s);
        
        for (let o of parsedSeq) {
            if (!observations.includes(o)) {
                setErrorMsg(`Lỗi: Quan sát '${o}' trong chuỗi không tồn tại trong tập quan sát!`);
                return false;
            }
        }
        if (parsedSeq.length === 0) {
            setErrorMsg("Chuỗi quan sát không được để trống!");
            return false;
        }

        setSequence(parsedSeq);
        
        // Chuẩn hoá số liệu từ chuỗi đầu vào (tránh lỗi gõ dấu chấm thập phân)
        const parsedPi = pi.map(p => { const val = parseFloat(p); return isNaN(val) ? 0 : val; });
        const parsedA = A.map(row => row.map(v => { const val = parseFloat(v); return isNaN(val) ? 0 : val; }));
        const parsedB = B.map(row => row.map(v => { const val = parseFloat(v); return isNaN(val) ? 0 : val; }));

        let steps = [];
        let V = []; 
        let P = []; 
        
        for(let t=0; t<parsedSeq.length; t++) {
            V.push(new Array(states.length).fill(null));
            P.push(new Array(states.length).fill(null));
        }

        steps.push({
            phase: 'INIT',
            t: -1, j: -1,
            desc: `Bắt đầu mô phỏng Viterbi.\nChuỗi quan sát: [${parsedSeq.join(', ')}]\nSố lượng trạng thái: ${states.length}`,
            V: JSON.parse(JSON.stringify(V)),
            P: JSON.parse(JSON.stringify(P))
        });

        // t = 0
        let obs_0 = parsedSeq[0];
        let obs_idx_0 = observations.indexOf(obs_0);
        
        for (let j = 0; j < states.length; j++) {
            let prob = parsedPi[j] * parsedB[j][obs_idx_0];
            V[0][j] = prob;

            steps.push({
                phase: 'CALC', t: 0, j: j,
                desc: `--- Bước t = 0, Trạng thái: ${states[j]} ---\nQuan sát: ${obs_0}\nV(0, ${states[j]}) = π(${states[j]}) × B(${states[j]} → ${obs_0})\n= ${parsedPi[j]} × ${parsedB[j][obs_idx_0]}\n= ${formatProb(prob)}`,
                V: JSON.parse(JSON.stringify(V)), P: JSON.parse(JSON.stringify(P))
            });
        }

        // t > 0
        for (let t = 1; t < parsedSeq.length; t++) {
            let obs_t = parsedSeq[t];
            let obs_idx = observations.indexOf(obs_t);

            for (let j = 0; j < states.length; j++) {
                let max_val = -1;
                let best_prev = -1;
                let calcs = [];

                for (let i = 0; i < states.length; i++) {
                    let trans_p = V[t-1][i] * parsedA[i][j];
                    calcs.push(`Từ ${states[i]}: ${formatProb(V[t-1][i])} × ${parsedA[i][j]} = ${formatProb(trans_p)}`);
                    if (trans_p > max_val) {
                        max_val = trans_p;
                        best_prev = i;
                    }
                }

                let emission_p = parsedB[j][obs_idx];
                let final_p = max_val * emission_p;
                
                V[t][j] = final_p;
                P[t][j] = best_prev;

                steps.push({
                    phase: 'CALC', t: t, j: j, best_prev: best_prev,
                    desc: `--- Bước t = ${t}, Trạng thái: ${states[j]} ---\nQuan sát: ${obs_t}\n1. Tìm MAX từ cột trước (t-${t-1}):\n  ${calcs.join('\n  ')}\n  => Max là ${formatProb(max_val)} (từ ${states[best_prev]})\n2. Nhân phát xạ B(${states[j]} → ${obs_t}) = ${emission_p}\nV(${t}, ${states[j]}) = ${formatProb(max_val)} × ${emission_p} = ${formatProb(final_p)}`,
                    V: JSON.parse(JSON.stringify(V)), P: JSON.parse(JSON.stringify(P))
                });
            }
        }

        // Kết thúc, tìm max cuối
        let final_max = -1;
        let best_final_state = -1;
        for (let j = 0; j < states.length; j++) {
            if (V[parsedSeq.length - 1][j] > final_max) {
                final_max = V[parsedSeq.length - 1][j];
                best_final_state = j;
            }
        }

        let fullPath = new Array(parsedSeq.length).fill(null);
        fullPath[parsedSeq.length - 1] = best_final_state;

        steps.push({
            phase: 'BACKTRACK_START', t: parsedSeq.length - 1, j: best_final_state,
            desc: `Đã tính xong bảng V.\nTìm max ở cột cuối (t=${parsedSeq.length - 1}):\nTrạng thái kết thúc tốt nhất là '${states[best_final_state]}' với xác suất = ${formatProb(final_max)}.\nBắt đầu truy vết ngược...`,
            V: JSON.parse(JSON.stringify(V)), P: JSON.parse(JSON.stringify(P)),
            path: [...fullPath]
        });

        // Backtracking
        let curr_state = best_final_state;
        for (let t = parsedSeq.length - 1; t > 0; t--) {
            let prev_state = P[t][curr_state];
            fullPath[t-1] = prev_state;
            
            steps.push({
                phase: 'BACKTRACK', t: t, j: curr_state, prev_j: prev_state,
                desc: `Truy vết tại t=${t}:\nTại trạng thái '${states[curr_state]}', pointer trỏ về '${states[prev_state]}' ở t=${t-1}.`,
                V: JSON.parse(JSON.stringify(V)), P: JSON.parse(JSON.stringify(P)),
                path: [...fullPath]
            });
            curr_state = prev_state;
        }

        let finalPathNames = fullPath.map(idx => states[idx]).join(' ➔ ');
        steps.push({
            phase: 'DONE', t: 0, j: curr_state,
            desc: `🎉 Hoàn tất!\nChuỗi trạng thái ẩn có xác suất cao nhất sinh ra chuỗi quan sát là:\n\n${finalPathNames}`,
            V: JSON.parse(JSON.stringify(V)), P: JSON.parse(JSON.stringify(P)),
            path: [...fullPath],
            finalStr: finalPathNames
        });

        setSimSteps(steps);
        setCurrentStepIdx(0);
        setIsPlaying(false);
        setErrorMsg("");
        setView('simulation');
        return true;
    };

    useEffect(() => {
        if (isPlaying) {
            timerRef.current = setTimeout(() => {
                if (currentStepIdx < simSteps.length - 1) {
                    setCurrentStepIdx(prev => prev + 1);
                } else {
                    setIsPlaying(false);
                }
            }, 1500);
        }
        return () => clearTimeout(timerRef.current);
    }, [isPlaying, currentStepIdx, simSteps.length]);

    const renderConfigView = () => (
        <div className="space-y-6 animate-fade-in pb-12">
            {errorMsg && (
                <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-4 rounded">
                    <p>{errorMsg}</p>
                </div>
            )}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200">
                    <h3 className="font-semibold text-lg text-slate-800 mb-4 border-b pb-2">1. Định nghĩa không gian</h3>
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Các Trạng thái Ẩn (cách nhau dấu phẩy)</label>
                            <input type="text" value={statesStr} onChange={e => setStatesStr(e.target.value)} onBlur={handleUpdateLists}
                                className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Các Quan sát (cách nhau dấu phẩy)</label>
                            <input type="text" value={obsStr} onChange={e => setObsStr(e.target.value)} onBlur={handleUpdateLists}
                                className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" />
                        </div>
                    </div>
                </div>

                <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200">
                    <h3 className="font-semibold text-lg text-slate-800 mb-4 border-b pb-2">4. Chuỗi quan sát cần dự đoán</h3>
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Chuỗi các quan sát theo thời gian (cách nhau dấu phẩy)</label>
                            <textarea value={seqStr} onChange={e => setSeqStr(e.target.value)} rows="3"
                                className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"></textarea>
                            <p className="text-xs text-slate-500 mt-1">Phải nằm trong tập Quan sát đã định nghĩa ở mục 1.</p>
                        </div>
                    </div>
                    <div className="mt-4">
                        <button onClick={generateSimulation} className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg flex justify-center items-center gap-2 transition-colors">
                            <IconPlayCircle />
                            Bắt đầu Mô phỏng Quy hoạch động
                        </button>
                    </div>
                </div>
            </div>

            <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200">
                <h3 className="font-semibold text-lg text-slate-800 mb-4 border-b pb-2">2. Ma trận Xác suất Ban đầu (π) & Chuyển trạng thái (A)</h3>
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 overflow-x-auto">
                    <div className="col-span-1">
                        <h4 className="text-sm font-bold text-slate-600 mb-2">Pi (π) - Initial</h4>
                        <table className="min-w-full divide-y divide-slate-200 text-sm">
                            <tbody>
                                {states.map((s, i) => (
                                    <tr key={`pi-${i}`}>
                                        <td className="py-2 pr-3 font-medium text-slate-700 bg-slate-50">{s}</td>
                                        <td className="py-2">
                                            <input type="text" inputMode="decimal" value={pi[i] !== undefined ? pi[i] : ""} 
                                                onChange={e => { let newPi = [...pi]; newPi[i] = e.target.value; setPi(newPi); }}
                                                className="w-24 px-2 py-1 border rounded" />
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    <div className="col-span-2 overflow-x-auto">
                        <h4 className="text-sm font-bold text-slate-600 mb-2">Ma trận A (Từ hàng \rightarrow Cột)</h4>
                        <table className="min-w-full divide-y divide-slate-200 text-sm">
                            <thead>
                                <tr>
                                    <th className="bg-slate-50"></th>
                                    {states.map(s => <th key={`ah-${s}`} className="px-2 py-2 text-center bg-slate-50">{s}</th>)}
                                </tr>
                            </thead>
                            <tbody>
                                {states.map((sRow, i) => (
                                    <tr key={`a-row-${i}`}>
                                        <td className="py-2 pr-3 font-medium text-slate-700 bg-slate-50 whitespace-nowrap">{sRow}</td>
                                        {states.map((sCol, j) => (
                                            <td key={`a-cell-${i}-${j}`} className="py-2 px-1 text-center">
                                                <input type="text" inputMode="decimal" value={A[i] && A[i][j] !== undefined ? A[i][j] : ""} 
                                                    onChange={e => {
                                                        let newA = A.map(row => [...row]);
                                                        newA[i][j] = e.target.value;
                                                        setA(newA);
                                                    }}
                                                    className="w-20 px-2 py-1 border rounded text-center" />
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200 overflow-x-auto">
                <h3 className="font-semibold text-lg text-slate-800 mb-4 border-b pb-2">3. Ma trận Xác suất Phát xạ (B)</h3>
                <p className="text-sm text-slate-500 mb-2">Trạng thái (hàng) sinh ra Quan sát (cột)</p>
                <table className="min-w-full divide-y divide-slate-200 text-sm">
                    <thead>
                        <tr>
                            <th className="bg-slate-50"></th>
                            {observations.map(o => <th key={`bh-${o}`} className="px-2 py-2 text-center bg-slate-50">{o}</th>)}
                        </tr>
                    </thead>
                    <tbody>
                        {states.map((s, i) => (
                            <tr key={`b-row-${i}`}>
                                <td className="py-2 pr-3 font-medium text-slate-700 bg-slate-50 whitespace-nowrap">{s}</td>
                                {observations.map((o, j) => (
                                    <td key={`b-cell-${i}-${j}`} className="py-2 px-1 text-center">
                                        <input type="text" inputMode="decimal" value={B[i] && B[i][j] !== undefined ? B[i][j] : ""} 
                                            onChange={e => {
                                                let newB = B.map(row => [...row]);
                                                newB[i][j] = e.target.value;
                                                setB(newB);
                                            }}
                                            className="w-20 px-2 py-1 border rounded text-center" />
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );

    const renderSimulationView = () => {
        if (simSteps.length === 0) return null;
        const stepData = simSteps[currentStepIdx];
        const V = stepData.V;
        const P = stepData.P;

        // Render classes for Table
        const getCellClass = (t, j) => {
            let base = "border px-3 py-3 text-center align-middle relative transition-all duration-300 ease-in-out ";
            const inPath = stepData.path && stepData.path[t] === j;

            if (stepData.phase === 'DONE' && inPath) return base + "bg-green-500 text-white font-bold border-green-600 shadow-md z-10";
            if (stepData.phase === 'CALC') {
                if (stepData.t === t && stepData.j === j) return base + "bg-yellow-100 border-yellow-400 font-bold shadow-inner ring-2 ring-yellow-400 z-10";
                if (stepData.t === t - 1) {
                    if (stepData.best_prev === j) return base + "bg-blue-200 border-blue-500 font-semibold";
                    return base + "bg-slate-100 opacity-60";
                }
            } else if (stepData.phase === 'BACKTRACK_START' || stepData.phase === 'BACKTRACK') {
                if (stepData.t === t && stepData.j === j) return base + "bg-green-200 border-green-500 font-bold ring-2 ring-green-500";
                if (stepData.phase === 'BACKTRACK' && stepData.t - 1 === t && stepData.prev_j === j) return base + "bg-green-100 border-green-400 border-dashed border-2";
                if (inPath && t >= stepData.t) return base + "bg-green-200 border-green-500 font-bold";
            }
            return base + "bg-white border-slate-200";
        };

        // --- Render Graph Elements ---
        const paddingX = 80;
        const paddingY = 60;
        const colWidth = 140;
        const rowHeight = 80;
        const graphWidth = Math.max(sequence.length * colWidth + paddingX * 2, 600);
        const graphHeight = states.length * rowHeight + paddingY * 1.5;

        const getX = (t) => paddingX + t * colWidth;
        const getY = (j) => paddingY + j * rowHeight;

        let graphEdges = [];
        let graphNodes = [];

        // Build Edges
        for (let t = 1; t < sequence.length; t++) {
            for (let j = 0; j < states.length; j++) {
                for (let i = 0; i < states.length; i++) {
                    const x1 = getX(t - 1);
                    const y1 = getY(i);
                    const x2 = getX(t);
                    const y2 = getY(j);

                    let color = "#e2e8f0"; // default: slate-200
                    let strokeW = 1.5;
                    let dash = "4 4";
                    let marker = "url(#arrowhead)";
                    let opacity = 0.3;
                    let zIndex = 0;

                    const isFinalPath = stepData.path && stepData.path[t - 1] === i && stepData.path[t] === j && stepData.path[t] !== null;

                    if (isFinalPath) {
                        color = "#22c55e"; // green
                        strokeW = 3;
                        dash = "none";
                        marker = "url(#arrowhead-green)";
                        opacity = 1;
                        zIndex = 10;
                    } else if (stepData.phase === 'CALC' && stepData.t === t && stepData.j === j) {
                        if (stepData.best_prev === i) {
                            color = "#3b82f6"; // blue
                            strokeW = 3;
                            dash = "none";
                            marker = "url(#arrowhead-blue)";
                            opacity = 1;
                            zIndex = 5;
                        } else {
                            color = "#f59e0b"; // amber
                            strokeW = 2;
                            dash = "none";
                            marker = "url(#arrowhead-amber)";
                            opacity = 0.8;
                            zIndex = 4;
                        }
                    } else if (t < stepData.t || (t === stepData.t && j < stepData.j) || (stepData.phase !== 'INIT' && stepData.phase !== 'CALC')) {
                        if (P[t][j] === i) {
                            color = "#94a3b8"; // slate-400
                            strokeW = 2;
                            dash = "none";
                            marker = "url(#arrowhead-slate)";
                            opacity = 0.6;
                            zIndex = 2;
                        }
                    }

                    graphEdges.push({ x1, y1, x2, y2, color, strokeW, dash, marker, opacity, zIndex, key: `e-${t}-${i}-${j}` });
                }
            }
        }
        graphEdges.sort((a, b) => a.zIndex - b.zIndex);

        // Build Nodes
        for (let t = 0; t < sequence.length; t++) {
            for (let j = 0; j < states.length; j++) {
                const x = getX(t);
                const y = getY(j);

                let fill = "#f8fafc";
                let stroke = "#cbd5e1";
                let strokeW = 2;
                let textFill = "#64748b";

                const inPath = stepData.path && stepData.path[t] === j && stepData.path[t] !== null;
                const isCurrent = stepData.t === t && stepData.j === j;

                if (inPath) {
                    fill = "#22c55e"; stroke = "#16a34a"; textFill = "#ffffff";
                } else if (isCurrent && stepData.phase === 'CALC') {
                    fill = "#fef08a"; stroke = "#eab308"; strokeW = 3; textFill = "#854d0e";
                } else if (isCurrent && stepData.phase.includes('BACKTRACK')) {
                    fill = "#bbf7d0"; stroke = "#22c55e"; strokeW = 3; textFill = "#166534";
                } else if (V[t][j] !== null && (t < stepData.t || (t === stepData.t && j < stepData.j) || stepData.phase !== 'CALC')) {
                    fill = "#eff6ff"; stroke = "#93c5fd"; textFill = "#1e3a8a";
                }

                graphNodes.push({ x, y, fill, stroke, strokeW, textFill, label: states[j].substring(0, 3), val: V[t][j], key: `n-${t}-${j}` });
            }
        }

        return (
            <div className="flex flex-col gap-4">
                {/* Thanh điều khiển */}
                <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 flex flex-wrap justify-between items-center gap-4 sticky top-20 z-40">
                    <div className="flex items-center gap-2 bg-slate-100 p-1.5 rounded-lg">
                        <button onClick={() => { setIsPlaying(false); setCurrentStepIdx(0); }} className="p-2 hover:bg-slate-200 rounded text-slate-600" title="Về đầu">
                            <IconReset />
                        </button>
                        <button onClick={() => { setIsPlaying(false); setCurrentStepIdx(Math.max(0, currentStepIdx - 1)); }} disabled={currentStepIdx === 0} className={`p-2 rounded ${currentStepIdx === 0 ? 'text-slate-300' : 'text-slate-700 hover:bg-slate-200'}`}>
                            <IconStepBack />
                        </button>
                        <button onClick={() => setIsPlaying(!isPlaying)} className={`p-2 px-4 rounded font-medium flex items-center gap-1 ${isPlaying ? 'bg-amber-100 text-amber-700' : 'bg-blue-600 text-white hover:bg-blue-700'}`}>
                            {isPlaying ? <><IconPause /> Tạm dừng</> : <><IconPlay /> Tự động chạy</>}
                        </button>
                        <button onClick={() => { setIsPlaying(false); setCurrentStepIdx(Math.min(simSteps.length - 1, currentStepIdx + 1)); }} disabled={currentStepIdx === simSteps.length - 1} className={`p-2 rounded ${currentStepIdx === simSteps.length - 1 ? 'text-slate-300' : 'text-slate-700 hover:bg-slate-200'}`}>
                            <IconStepForward />
                        </button>
                    </div>

                    <div className="text-sm font-medium text-slate-600 flex items-center gap-2">
                        <span className="bg-slate-200 px-3 py-1 rounded-full">Bước {currentStepIdx + 1} / {simSteps.length}</span>
                        <span className={`px-3 py-1 rounded-full text-xs font-bold ${stepData.phase === 'DONE' ? 'bg-green-100 text-green-700' : stepData.phase.includes('BACKTRACK') ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'}`}>
                            {stepData.phase === 'INIT' ? 'KHỞI TẠO' : stepData.phase === 'CALC' ? 'QUY HOẠCH ĐỘNG' : stepData.phase === 'DONE' ? 'HOÀN THÀNH' : 'TRUY VẾT NGƯỢC'}
                        </span>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-0">
                    <div className="lg:col-span-2 flex flex-col gap-4">
                        {/* Khung Graph */}
                        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-4 overflow-x-auto relative">
                            <h3 className="font-semibold text-slate-800 mb-2">Mạng tinh thể (Trellis Diagram)</h3>
                            <svg width={graphWidth} height={graphHeight} className="block">
                                <defs>
                                    <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="21" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="#e2e8f0" /></marker>
                                    <marker id="arrowhead-slate" markerWidth="8" markerHeight="6" refX="21" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="#94a3b8" /></marker>
                                    <marker id="arrowhead-blue" markerWidth="8" markerHeight="6" refX="21" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="#3b82f6" /></marker>
                                    <marker id="arrowhead-amber" markerWidth="8" markerHeight="6" refX="21" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="#f59e0b" /></marker>
                                    <marker id="arrowhead-green" markerWidth="8" markerHeight="6" refX="21" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="#22c55e" /></marker>
                                </defs>
                                
                                {/* Labels Y (Trạng thái) */}
                                {states.map((s, j) => (
                                    <text key={`ly-${j}`} x={paddingX - 30} y={getY(j)} fontSize="12" fontWeight="bold" fill="#475569" textAnchor="end" dominantBaseline="middle">{s}</text>
                                ))}
                                
                                {/* Labels X (Quan sát) */}
                                {sequence.map((obs, t) => (
                                    <g key={`lx-${t}`}>
                                        <text x={getX(t)} y={25} fontSize="11" fill="#64748b" textAnchor="middle">t = {t}</text>
                                        <text x={getX(t)} y={40} fontSize="12" fontWeight="bold" fill="#1e40af" textAnchor="middle">{obs}</text>
                                    </g>
                                ))}

                                {/* Edges */}
                                {graphEdges.map(e => (
                                    <line key={e.key} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} stroke={e.color} strokeWidth={e.strokeW} strokeDasharray={e.dash} markerEnd={e.marker} opacity={e.opacity} className="transition-all duration-300" />
                                ))}

                                {/* Nodes */}
                                {graphNodes.map(n => (
                                    <g key={n.key} className="transition-all duration-300">
                                        <circle cx={n.x} cy={n.y} r="16" fill={n.fill} stroke={n.stroke} strokeWidth={n.strokeW} />
                                        <text x={n.x} y={n.y} fontSize="10" fill={n.textFill} fontWeight="bold" textAnchor="middle" dominantBaseline="central">{n.label}</text>
                                        {n.val !== null && (
                                            <text x={n.x} y={n.y + 26} fontSize="10" fill="#64748b" textAnchor="middle" className="font-mono">{formatProb(n.val)}</text>
                                        )}
                                    </g>
                                ))}
                            </svg>
                        </div>

                        {/* Khung Bảng */}
                        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-4 overflow-auto">
                            <h3 className="font-semibold text-slate-800 mb-4 sticky left-0">Bảng Quy Hoạch Động (Ma trận V)</h3>
                            <table className="min-w-full border-collapse">
                                <thead>
                                    <tr>
                                        <th className="border border-slate-300 bg-slate-100 px-4 py-2 sticky left-0 z-20 shadow-[2px_0_5px_-2px_rgba(0,0,0,0.1)]">Trạng Thái \ t</th>
                                        {sequence.map((obs, t) => (
                                            <th key={`th-${t}`} className="border border-slate-300 bg-slate-50 px-4 py-2 min-w-[140px]">
                                                <div className="text-xs text-slate-500 font-normal">t = {t}</div>
                                                <div className="font-bold text-blue-800">{obs}</div>
                                            </th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {states.map((stateName, j) => (
                                        <tr key={`tr-${j}`}>
                                            <td className="border border-slate-300 bg-slate-100 px-4 py-3 font-semibold text-slate-700 sticky left-0 z-10 shadow-[2px_0_5px_-2px_rgba(0,0,0,0.1)]">
                                                {stateName}
                                            </td>
                                            {sequence.map((_, t) => (
                                                <td key={`td-${t}-${j}`} className={getCellClass(t, j)}>
                                                    {V[t][j] !== null ? (
                                                        <div className="flex flex-col items-center justify-center h-full">
                                                            <span className="font-mono">{formatProb(V[t][j])}</span>
                                                            {P[t][j] !== null && (
                                                                <span className="inline-block text-[0.8rem] text-slate-500 ml-1 mt-1 bg-slate-100 px-1.5 rounded border">
                                                                    ← {states[P[t][j]]}
                                                                </span>
                                                            )}
                                                        </div>
                                                    ) : (
                                                        <span className="text-slate-300">-</span>
                                                    )}
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* Khung Diễn giải */}
                    <div className="bg-slate-800 text-slate-100 rounded-xl shadow-sm border border-slate-700 p-5 flex flex-col overflow-hidden max-h-[800px]">
                        <h3 className="font-semibold text-white mb-3 flex items-center gap-2 border-b border-slate-600 pb-2">
                            <IconSettings /> Chi tiết tính toán
                        </h3>
                        <div className="flex-1 overflow-auto font-mono text-sm whitespace-pre-wrap leading-relaxed">
                            {stepData.desc}
                        </div>
                        {stepData.phase === 'DONE' && (
                            <div className="mt-4 p-4 bg-green-600/20 border border-green-500 rounded-lg">
                                <h4 className="font-bold text-green-400 mb-2">Kết quả Cuối cùng:</h4>
                                <p className="text-lg font-bold text-white leading-tight">{stepData.finalStr}</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="min-h-screen bg-slate-50 font-sans flex flex-col text-slate-900">
            {/* Header */}
            <header className="bg-white shadow-sm border-b border-slate-200 px-6 py-4 flex items-center justify-between sticky top-0 z-50">
                <div>
                    <h1 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
                        Mô phỏng Viterbi <span className="bg-blue-100 text-blue-700 text-xs px-2 py-1 rounded-md align-middle">HMM</span>
                    </h1>
                    <p className="text-sm text-slate-500 mt-1">Thuật toán quy hoạch động tìm chuỗi trạng thái ẩn tối ưu</p>
                </div>
                <div className="flex bg-slate-100 p-1 rounded-lg">
                    <button onClick={() => { setView('config'); setIsPlaying(false); }} className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${view === 'config' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-600 hover:bg-slate-200'}`}>
                        1. Cấu hình Model
                    </button>
                    <button onClick={() => { if(simSteps.length > 0) setView('simulation'); else generateSimulation(); }} className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${view === 'simulation' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-600 hover:bg-slate-200'}`}>
                        2. Chạy Mô phỏng
                    </button>
                </div>
            </header>

            {/* Main Content */}
            <main className="flex-1 p-6 max-w-[1400px] mx-auto w-full">
                {view === 'config' ? renderConfigView() : renderSimulationView()}
            </main>
        </div>
    );
}
