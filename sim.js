/**
 * Reaction-Diffusion WebGL2 / WebGL1 Robust Implementation
 * Based on Karl Sims Gray-Scott model tutorial.
 */

(function () {
  'use strict';

  const canvas = document.getElementById('rdCanvas');

  // Try WebGL2 first (preferred for high-precision float FBO)
  let gl = canvas.getContext('webgl2', { antialias: false, depth: false, alpha: false, preserveDrawingBuffer: false, powerPreference: 'high-performance' });
  let isWebGL2 = !!gl;

  let internalFormat, format, type;

  if (gl) {
    const extColorFloat = gl.getExtension('EXT_color_buffer_float');
    // In WebGL2, EXT_color_buffer_float enables rendering to RGBA32F, RGBA16F, R32F, etc.
    // Try RGBA16F or RGBA32F
    if (extColorFloat) {
      internalFormat = gl.RGBA16F; // RGBA16F is widely supported and faster/stable
      format = gl.RGBA;
      type = gl.HALF_FLOAT;
    } else {
      internalFormat = gl.RGBA;
      format = gl.RGBA;
      type = gl.UNSIGNED_BYTE;
      console.warn('EXT_color_buffer_float não suportado no WebGL2. Usando fallback 8-bit.');
    }
  } else {
    // WebGL1 fallback
    gl = canvas.getContext('webgl', { antialias: false, depth: false, alpha: false, powerPreference: 'high-performance' }) ||
         canvas.getContext('experimental-webgl');
    if (!gl) {
      alert('Seu navegador não suporta WebGL.');
      return;
    }
    const floatExt = gl.getExtension('OES_texture_float');
    const halfFloatExt = gl.getExtension('OES_texture_half_float');
    
    internalFormat = gl.RGBA;
    format = gl.RGBA;
    if (floatExt) {
      type = gl.FLOAT;
    } else if (halfFloatExt) {
      type = halfFloatExt.HALF_FLOAT_OES;
    } else {
      type = gl.UNSIGNED_BYTE;
    }
  }

  console.log(`WebGL initialized: isWebGL2=${isWebGL2}, type=${type}, internalFormat=${internalFormat}`);

  // Presets
  const PRESETS = {
    coral:    { F: 0.0545, k: 0.0620, speed: 18 },
    mitosis:  { F: 0.0367, k: 0.0649, speed: 20 },
    solitons: { F: 0.0300, k: 0.0620, speed: 22 },
    waves:    { F: 0.0140, k: 0.0540, speed: 18 },
    spots:    { F: 0.0400, k: 0.0600, speed: 18 },
    chaos:    { F: 0.0260, k: 0.0550, speed: 18 },
    custom:   { F: 0.0545, k: 0.0620, speed: 18 }
  };

  const simParams = {
    F: 0.0545,
    k: 0.0620,
    Da: 1.0,
    Db: 0.5,
    speed: 18,
    brushRadius: 24,
    palette: 0,
    paused: false
  };

  let simWidth = 512;
  let simHeight = 512;

  // Quad Geometry
  const quadBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,
     1, -1,
    -1,  1,
    -1,  1,
     1, -1,
     1,  1
  ]), gl.STATIC_DRAW);

  function createShader(gl, type, source) {
    const s = gl.createShader(type);
    gl.shaderSource(s, source);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error('Shader compile error:', gl.getShaderInfoLog(s), source);
      return null;
    }
    return s;
  }

  function createProgram(gl, vsSrc, fsSrc) {
    const vs = createShader(gl, gl.VERTEX_SHADER, vsSrc);
    const fs = createShader(gl, gl.FRAGMENT_SHADER, fsSrc);
    if (!vs || !fs) return null;
    const p = gl.createProgram();
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(p));
      return null;
    }
    return p;
  }

  // Shaders (WebGL1 / WebGL2 Compatible)
  const vsQuad = `
    precision highp float;
    attribute vec2 a_position;
    varying vec2 v_uv;
    void main() {
      v_uv = (a_position + 1.0) * 0.5;
      gl_Position = vec4(a_position, 0.0, 1.0);
    }
  `;

  // Simulation Fragment Shader
  const fsSim = `
    precision highp float;
    varying vec2 v_uv;
    uniform sampler2D u_state;
    uniform vec2 u_resolution;
    uniform float u_F;
    uniform float u_K;
    uniform float u_Da;
    uniform float u_Db;
    uniform float u_dt;
    uniform vec3 u_brush; // x, y, radius in normalized UV

    void main() {
      vec2 step = 1.0 / u_resolution;
      vec2 current = texture2D(u_state, v_uv).rg;
      float a = current.r;
      float b = current.g;

      // Karl Sims 3x3 Convolution Stencil
      // Direct orthogonal neighbors (weight 0.2)
      // Diagonals (weight 0.05)
      // Center (weight -1.0)
      vec2 lap = vec2(0.0);
      lap += texture2D(u_state, v_uv + vec2( 0.0,      step.y)).rg * 0.2;
      lap += texture2D(u_state, v_uv + vec2( 0.0,     -step.y)).rg * 0.2;
      lap += texture2D(u_state, v_uv + vec2(-step.x,   0.0   )).rg * 0.2;
      lap += texture2D(u_state, v_uv + vec2( step.x,   0.0   )).rg * 0.2;

      lap += texture2D(u_state, v_uv + vec2(-step.x,   step.y)).rg * 0.05;
      lap += texture2D(u_state, v_uv + vec2( step.x,   step.y)).rg * 0.05;
      lap += texture2D(u_state, v_uv + vec2(-step.x,  -step.y)).rg * 0.05;
      lap += texture2D(u_state, v_uv + vec2( step.x,  -step.y)).rg * 0.05;

      lap -= current;

      float abb = a * b * b;
      float dA = (u_Da * lap.r - abb + u_F * (1.0 - a)) * u_dt;
      float dB = (u_Db * lap.g + abb - (u_K + u_F) * b) * u_dt;

      float nextA = clamp(a + dA, 0.0, 1.0);
      float nextB = clamp(b + dB, 0.0, 1.0);

      // Interactive mouse brush
      if (u_brush.z > 0.0) {
        float aspect = u_resolution.x / u_resolution.y;
        vec2 diff = (v_uv - u_brush.xy) * vec2(aspect, 1.0);
        float dist = length(diff);
        if (dist < u_brush.z) {
          float str = 1.0 - smoothstep(0.0, u_brush.z, dist);
          nextB = clamp(nextB + 0.9 * str, 0.0, 1.0);
          nextA = clamp(nextA - 0.7 * str, 0.0, 1.0);
        }
      }

      gl_FragColor = vec4(nextA, nextB, 0.0, 1.0);
    }
  `;

  // Render Display Shader
  const fsRender = `
    precision highp float;
    varying vec2 v_uv;
    uniform sampler2D u_state;
    uniform int u_palette;

    void main() {
      vec2 state = texture2D(u_state, v_uv).rg;
      float a = state.r;
      float b = state.g;

      // Enhance dynamic range: B typically settles around 0.15 - 0.40 in pattern zones
      float v = clamp(b * 3.2, 0.0, 1.0);

      vec3 color = vec3(0.0);

      if (u_palette == 0) {
        // Cyan / Neon Bioluminescent
        vec3 c0 = vec3(0.04, 0.05, 0.09);
        vec3 c1 = vec3(0.0, 0.55, 1.0);
        vec3 c2 = vec3(0.0, 0.95, 1.0);
        vec3 c3 = vec3(0.85, 1.0, 1.0);
        if (v < 0.15) color = mix(c0, c1, v / 0.15);
        else if (v < 0.60) color = mix(c1, c2, (v - 0.15) / 0.45);
        else color = mix(c2, c3, (v - 0.60) / 0.40);
      } else if (u_palette == 1) {
        // Cyberpunk Violet / Sunset
        vec3 c0 = vec3(0.06, 0.03, 0.10);
        vec3 c1 = vec3(0.65, 0.12, 0.85);
        vec3 c2 = vec3(1.0, 0.25, 0.55);
        vec3 c3 = vec3(1.0, 0.95, 0.65);
        if (v < 0.20) color = mix(c0, c1, v / 0.20);
        else if (v < 0.65) color = mix(c1, c2, (v - 0.20) / 0.45);
        else color = mix(c2, c3, (v - 0.65) / 0.35);
      } else if (u_palette == 2) {
        // Esmeralda Deep Sea
        vec3 c0 = vec3(0.02, 0.06, 0.06);
        vec3 c1 = vec3(0.0, 0.55, 0.45);
        vec3 c2 = vec3(0.1, 1.0, 0.65);
        vec3 c3 = vec3(0.9, 1.0, 0.9);
        if (v < 0.20) color = mix(c0, c1, v / 0.20);
        else if (v < 0.65) color = mix(c1, c2, (v - 0.20) / 0.45);
        else color = mix(c2, c3, (v - 0.65) / 0.35);
      } else if (u_palette == 3) {
        // Monochromatic Editorial (B&W)
        float g = pow(v, 0.9);
        color = vec3(g);
      } else {
        // Magma Elétrico
        vec3 c0 = vec3(0.08, 0.02, 0.03);
        vec3 c1 = vec3(0.95, 0.20, 0.0);
        vec3 c2 = vec3(1.0, 0.70, 0.0);
        vec3 c3 = vec3(1.0, 1.0, 0.95);
        if (v < 0.20) color = mix(c0, c1, v / 0.20);
        else if (v < 0.65) color = mix(c1, c2, (v - 0.20) / 0.45);
        else color = mix(c2, c3, (v - 0.65) / 0.35);
      }

      // Subtle vignette
      vec2 coord = v_uv - 0.5;
      float vignette = 1.0 - dot(coord, coord) * 0.25;
      color *= vignette;

      gl_FragColor = vec4(color, 1.0);
    }
  `;

  // Seed Program
  const fsSeed = `
    precision highp float;
    varying vec2 v_uv;
    uniform int u_mode; // 0 = clear, 1 = center seed, 2 = random spots

    float hash(vec2 p) {
      p = fract(p * vec2(123.34, 456.21));
      p += dot(p, p + 45.32);
      return fract(p.x * p.y);
    }

    void main() {
      float a = 1.0;
      float b = 0.0;

      if (u_mode == 1) {
        // Prominent clusters across the banner so pattern is immediately visible everywhere
        vec2 c1 = abs(v_uv - vec2(0.50, 0.50));
        vec2 c2 = abs(v_uv - vec2(0.35, 0.45));
        vec2 c3 = abs(v_uv - vec2(0.65, 0.55));
        vec2 c4 = abs(v_uv - vec2(0.20, 0.50));
        vec2 c5 = abs(v_uv - vec2(0.80, 0.50));
        vec2 c6 = abs(v_uv - vec2(0.50, 0.25));
        vec2 c7 = abs(v_uv - vec2(0.50, 0.75));

        if ((c1.x < 0.035 && c1.y < 0.035) ||
            (c2.x < 0.030 && c2.y < 0.030) ||
            (c3.x < 0.030 && c3.y < 0.030) ||
            (c4.x < 0.025 && c4.y < 0.025) ||
            (c5.x < 0.025 && c5.y < 0.025) ||
            (c6.x < 0.025 && c6.y < 0.025) ||
            (c7.x < 0.025 && c7.y < 0.025)) {
          b = 1.0;
          a = 0.0;
        }
      } else if (u_mode == 2) {
        float r = hash(v_uv * 240.0);
        if (r > 0.975) {
          b = 1.0;
          a = 0.0;
        }
      }

      gl_FragColor = vec4(a, b, 0.0, 1.0);
    }
  `;

  const simProg = createProgram(gl, vsQuad, fsSim);
  const renderProg = createProgram(gl, vsQuad, fsRender);
  const seedProg = createProgram(gl, vsQuad, fsSeed);

  if (!simProg || !renderProg || !seedProg) {
    console.error('Falha ao compilar os programas WebGL.');
    return;
  }

  const simLoc = {
    pos: gl.getAttribLocation(simProg, 'a_position'),
    state: gl.getUniformLocation(simProg, 'u_state'),
    res: gl.getUniformLocation(simProg, 'u_resolution'),
    F: gl.getUniformLocation(simProg, 'u_F'),
    K: gl.getUniformLocation(simProg, 'u_K'),
    Da: gl.getUniformLocation(simProg, 'u_Da'),
    Db: gl.getUniformLocation(simProg, 'u_Db'),
    dt: gl.getUniformLocation(simProg, 'u_dt'),
    brush: gl.getUniformLocation(simProg, 'u_brush')
  };

  const renderLoc = {
    pos: gl.getAttribLocation(renderProg, 'a_position'),
    state: gl.getUniformLocation(renderProg, 'u_state'),
    palette: gl.getUniformLocation(renderProg, 'u_palette')
  };

  const seedLoc = {
    pos: gl.getAttribLocation(seedProg, 'a_position'),
    mode: gl.getUniformLocation(seedProg, 'u_mode')
  };

  // Ping-Pong Textures & Framebuffers
  let textures = [null, null];
  let framebuffers = [null, null];
  let currentRead = 0;

  function createFBOTexture(w, h, intFmt, fmt, tp) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texImage2D(gl.TEXTURE_2D, 0, intFmt, w, h, 0, fmt, tp, null);
    return tex;
  }

  function initPingPong(w, h) {
    // Determine working format once if not yet verified
    for (let i = 0; i < 2; i++) {
      if (textures[i]) gl.deleteTexture(textures[i]);
      if (framebuffers[i]) gl.deleteFramebuffer(framebuffers[i]);
    }

    let created = false;
    // Try current format first, then fallbacks
    const formatCandidates = isWebGL2 ? [
      { intFmt: gl.RGBA16F, fmt: gl.RGBA, tp: gl.HALF_FLOAT },
      { intFmt: gl.RGBA32F, fmt: gl.RGBA, tp: gl.FLOAT },
      { intFmt: gl.RGBA8,   fmt: gl.RGBA, tp: gl.UNSIGNED_BYTE }
    ] : [
      { intFmt: gl.RGBA, fmt: gl.RGBA, tp: type },
      { intFmt: gl.RGBA, fmt: gl.RGBA, tp: gl.UNSIGNED_BYTE }
    ];

    for (const cand of formatCandidates) {
      let valid = true;
      for (let i = 0; i < 2; i++) {
        textures[i] = createFBOTexture(w, h, cand.intFmt, cand.fmt, cand.tp);
        framebuffers[i] = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers[i]);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, textures[i], 0);
        const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
        if (status !== gl.FRAMEBUFFER_COMPLETE) {
          valid = false;
          break;
        }
      }
      if (valid) {
        internalFormat = cand.intFmt;
        format = cand.fmt;
        type = cand.tp;
        created = true;
        console.log('Framebuffer configured successfully with format:', cand);
        break;
      } else {
        for (let i = 0; i < 2; i++) {
          if (textures[i]) gl.deleteTexture(textures[i]);
          if (framebuffers[i]) gl.deleteFramebuffer(framebuffers[i]);
        }
      }
    }

    if (!created) {
      console.error('Falha ao criar Framebuffer compatível para a simulação.');
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  function seedTexture(mode) {
    gl.useProgram(seedProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(seedLoc.pos);
    gl.vertexAttribPointer(seedLoc.pos, 2, gl.FLOAT, false, 0, 0);
    gl.uniform1i(seedLoc.mode, mode);

    for (let i = 0; i < 2; i++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers[i]);
      gl.viewport(0, 0, simWidth, simHeight);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // Mouse & Touch Interaction
  let mouse = {
    x: 0.5,
    y: 0.5,
    down: false,
    hovering: false
  };

  function updateMouseCoord(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX !== undefined ? e.clientX : (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
    const clientY = e.clientY !== undefined ? e.clientY : (e.touches && e.touches[0] ? e.touches[0].clientY : 0);
    mouse.x = (clientX - rect.left) / rect.width;
    mouse.y = 1.0 - (clientY - rect.top) / rect.height; // WebGL UV flip
  }

  window.addEventListener('mousemove', (e) => {
    updateMouseCoord(e);
    mouse.hovering = true;
  });

  window.addEventListener('mousedown', (e) => {
    if (e.target.closest('#controlPanel') || e.target.closest('.hero-actions')) return;
    mouse.down = true;
    updateMouseCoord(e);
  });

  window.addEventListener('mouseup', () => {
    mouse.down = false;
  });

  window.addEventListener('touchmove', (e) => {
    updateMouseCoord(e);
    mouse.hovering = true;
    mouse.down = true;
  }, { passive: true });

  window.addEventListener('touchend', () => {
    mouse.down = false;
    mouse.hovering = false;
  });

  // Window resize & high-DPI scaling
  function resize() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;

    const baseRes = 512;
    const aspect = width / height;
    if (aspect >= 1.0) {
      simWidth = Math.round(baseRes * aspect);
      simHeight = baseRes;
    } else {
      simWidth = baseRes;
      simHeight = Math.round(baseRes / aspect);
    }

    initPingPong(simWidth, simHeight);
    seedTexture(2); // Iniciar por padrão com perturbação randômica
  }

  window.addEventListener('resize', resize);

  // UI Bindings
  const dispF = document.getElementById('dispF');
  const dispK = document.getElementById('dispK');
  const dispSpeed = document.getElementById('dispSpeed');
  const sliderF = document.getElementById('sliderF');
  const sliderK = document.getElementById('sliderK');
  const sliderSpeed = document.getElementById('sliderSpeed');
  const sliderBrush = document.getElementById('sliderBrush');
  const sliderValF = document.getElementById('sliderValF');
  const sliderValK = document.getElementById('sliderValK');
  const sliderValSpeed = document.getElementById('sliderValSpeed');
  const sliderValBrush = document.getElementById('sliderValBrush');
  const presetSelect = document.getElementById('presetSelect');
  const paletteSelect = document.getElementById('paletteSelect');
  const btnSeedCenter = document.getElementById('btnSeedCenter');
  const btnRandomize = document.getElementById('btnRandomize');
  const btnClear = document.getElementById('btnClear');
  const btnPause = document.getElementById('btnPause');
  const pauseText = document.getElementById('pauseText');
  const btnTogglePanel = document.getElementById('btnTogglePanel');
  const controlPanel = document.getElementById('controlPanel');

  function syncUI() {
    dispF.textContent = simParams.F.toFixed(4);
    dispK.textContent = simParams.k.toFixed(4);
    dispSpeed.textContent = simParams.speed + 'x';
    sliderValF.textContent = simParams.F.toFixed(4);
    sliderValK.textContent = simParams.k.toFixed(4);
    sliderValSpeed.textContent = simParams.speed + 'x';
    sliderValBrush.textContent = simParams.brushRadius + 'px';

    sliderF.value = simParams.F;
    sliderK.value = simParams.k;
    sliderSpeed.value = simParams.speed;
    sliderBrush.value = simParams.brushRadius;
  }

  sliderF.addEventListener('input', (e) => {
    simParams.F = parseFloat(e.target.value);
    presetSelect.value = 'custom';
    syncUI();
  });

  sliderK.addEventListener('input', (e) => {
    simParams.k = parseFloat(e.target.value);
    presetSelect.value = 'custom';
    syncUI();
  });

  sliderSpeed.addEventListener('input', (e) => {
    simParams.speed = parseInt(e.target.value, 10);
    syncUI();
  });

  sliderBrush.addEventListener('input', (e) => {
    simParams.brushRadius = parseInt(e.target.value, 10);
    syncUI();
  });

  presetSelect.addEventListener('change', (e) => {
    const val = e.target.value;
    if (PRESETS[val]) {
      simParams.F = PRESETS[val].F;
      simParams.k = PRESETS[val].k;
      simParams.speed = PRESETS[val].speed;
      syncUI();
      seedTexture(1);
    }
  });

  paletteSelect.addEventListener('change', (e) => {
    simParams.palette = parseInt(e.target.value, 10);
  });

  btnSeedCenter.addEventListener('click', () => seedTexture(1));
  btnRandomize.addEventListener('click', () => seedTexture(2));
  btnClear.addEventListener('click', () => seedTexture(0));

  btnPause.addEventListener('click', () => {
    simParams.paused = !simParams.paused;
    pauseText.textContent = simParams.paused ? 'Continuar' : 'Pausar';
  });

  btnTogglePanel.addEventListener('click', () => {
    controlPanel.classList.toggle('minimized');
  });

  // Step Simulation
  function stepSimulation() {
    gl.useProgram(simProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(simLoc.pos);
    gl.vertexAttribPointer(simLoc.pos, 2, gl.FLOAT, false, 0, 0);

    gl.uniform2f(simLoc.res, simWidth, simHeight);
    gl.uniform1f(simLoc.F, simParams.F);
    gl.uniform1f(simLoc.K, simParams.k);
    gl.uniform1f(simLoc.Da, simParams.Da);
    gl.uniform1f(simLoc.Db, simParams.Db);
    gl.uniform1f(simLoc.dt, 1.0);

    const iters = simParams.speed;
    const brushNorm = simParams.brushRadius / Math.min(canvas.width, canvas.height);

    // Auto-perturbação periódica e orgânica após estabilização
    handleAutoPerturbation();

    for (let i = 0; i < iters; i++) {
      const readIdx = currentRead;
      const writeIdx = 1 - currentRead;

      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers[writeIdx]);
      gl.viewport(0, 0, simWidth, simHeight);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, textures[readIdx]);
      gl.uniform1i(simLoc.state, 0);

      // Mouse brush or auto-pulse brush on first substep
      if (i === 0 && (mouse.down || autoPulse.active)) {
        const bx = mouse.down ? mouse.x : autoPulse.x;
        const by = mouse.down ? mouse.y : autoPulse.y;
        const br = mouse.down ? brushNorm : autoPulse.radius;
        gl.uniform3f(simLoc.brush, bx, by, br);
      } else {
        gl.uniform3f(simLoc.brush, 0.0, 0.0, -1.0);
      }

      gl.drawArrays(gl.TRIANGLES, 0, 6);
      currentRead = writeIdx;
    }

    if (!mouse.down) {
      mouse.hovering = false;
    }
  }

  // Sistema de Auto-Perturbação Orgânica
  // Quando o usuário não interage e o padrão atinge estabilidade morfológica,
  // injeta pequenas perturbações químicas pontuais para gerar novos brotos de crescimento
  let lastUserActivity = Date.now();
  let lastAutoPulseTime = Date.now();
  const autoPulse = {
    active: false,
    x: 0.5,
    y: 0.5,
    radius: 0.03,
    framesLeft: 0
  };

  function registerUserActivity() {
    lastUserActivity = Date.now();
  }

  window.addEventListener('mousemove', registerUserActivity);
  window.addEventListener('mousedown', registerUserActivity);
  window.addEventListener('touchstart', registerUserActivity);

  function handleAutoPerturbation() {
    const now = Date.now();
    // Se o pulso atual estiver ativo, decrescer duração (dura 2-3 frames para semear de forma suave)
    if (autoPulse.active) {
      autoPulse.framesLeft--;
      if (autoPulse.framesLeft <= 0) {
        autoPulse.active = false;
      }
      return;
    }

    // Intervalo de pulso automático: a cada 9 a 15 segundos de calmaria
    const timeSinceLastPulse = now - lastAutoPulseTime;
    const timeSinceUser = now - lastUserActivity;

    if (timeSinceUser > 5000 && timeSinceLastPulse > 9000 + Math.random() * 6000) {
      lastAutoPulseTime = now;
      // Ponto aleatório bem distribuído, evitando apenas as bordas extremas
      autoPulse.x = 0.15 + Math.random() * 0.70;
      autoPulse.y = 0.15 + Math.random() * 0.70;
      autoPulse.radius = 0.02 + Math.random() * 0.025; // raio sutil e orgânico
      autoPulse.framesLeft = 2; // dura 2 frames de simulação
      autoPulse.active = true;
    }
  }

  // Render to Viewport
  function renderToScreen() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);

    gl.useProgram(renderProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(renderLoc.pos);
    gl.vertexAttribPointer(renderLoc.pos, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, textures[currentRead]);
    gl.uniform1i(renderLoc.state, 0);
    gl.uniform1i(renderLoc.palette, simParams.palette);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  function frame() {
    if (!simParams.paused) {
      stepSimulation();
    }
    renderToScreen();
    requestAnimationFrame(frame);
  }

  // Init
  resize();
  syncUI();
  seedTexture(2); // Inicia por padrão com perturbação randômica
  requestAnimationFrame(frame);

})();
