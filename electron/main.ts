import { app, BrowserWindow, ipcMain, dialog, Menu, protocol, net, shell, nativeTheme } from 'electron'
import * as path from 'path'
import * as fs from 'fs/promises'
import { existsSync, mkdirSync, readFileSync, writeFileSync, unlinkSync } from 'fs'
import { fileURLToPath, pathToFileURL } from 'url'
import { type ChildProcess } from 'child_process'
import * as os from 'os'
import { APP_VERSION, ASR_GITHUB_REPO } from './constants'
import { findBackupPath } from '../src/utils/fileUtils'


const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const CAPTIONS_JSON_SUFFIX = '.captions_json5'
const captions_json5_files = ['captions_json5', 'captions_json']
const srt_files = ['srt']
const MIME_TYPES: Record<string, string> = {
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.ogg': 'video/ogg',
  '.mp3': 'audio/mpeg',
  '.aac': 'audio/aac',
  '.wav': 'audio/wav',
  '.mov': 'video/quicktime',
  '.m4a': 'audio/mp4',
  '.flac': 'audio/flac'
}
const media_files = Object.keys(MIME_TYPES).map(ext => ext.substring(1))
const all_files = captions_json5_files.concat(srt_files, media_files)

/** If `filePath` is a known media type and a sibling `<stem>.captions_json5` exists, open that instead. */
function resolveOpenFilePathPreferSiblingCaptions(filePath: string): string {
  const resolved = path.resolve(filePath)
  const ext = path.extname(resolved).toLowerCase()
  if (!(ext in MIME_TYPES)) return resolved
  const sibling = path.join(path.dirname(resolved), `${path.basename(resolved, path.extname(resolved))}${CAPTIONS_JSON_SUFFIX}`)
  return existsSync(sibling) ? sibling : resolved
}

// Register custom protocols as privileged for media streaming
protocol.registerSchemesAsPrivileged([
  { scheme: 'media', privileges: { secure: true, standard: true, supportFetchAPI: true, stream: true, bypassCSP: false } }
])

// Log version on startup
console.log(`[main] ========================================`)
console.log(`[main] Caption Editor v${APP_VERSION}`)
console.log(`[main] Electron v${process.versions.electron}`)
console.log(`[main] Chrome v${process.versions.chrome}`)
console.log(`[main] Node v${process.versions.node}`)
console.log(`[main] Platform: ${process.platform}`)
console.log(`[main] ========================================`)

// Store security-scoped bookmarks for macOS
const fileBookmarks = new Map<string, Buffer>()

/** Helper: get the BrowserWindow that sent an IPC event, or the focused window as fallback. */
function windowForEvent(event: Electron.IpcMainEvent | Electron.IpcMainInvokeEvent): BrowserWindow | null {
  return BrowserWindow.fromWebContents(event.sender) || BrowserWindow.getFocusedWindow()
}

/** Helper: get the focused window (for menu clicks which have no event). */
function focusedWindow(): BrowserWindow | null {
  return BrowserWindow.getFocusedWindow() || BrowserWindow.getAllWindows()[0] || null
}

function createMenu() {
  const isMac = process.platform === 'darwin'

  const template: Electron.MenuItemConstructorOptions[] = [
    // App menu (macOS only)
    ...(isMac ? [{
      label: app.name,
      submenu: [
        { role: 'about' as const },
        { type: 'separator' as const },
        { role: 'hide' as const },
        { role: 'hideOthers' as const },
        { role: 'unhide' as const },
        { type: 'separator' as const },
        { role: 'quit' as const }
      ]
    }] : []),

    // File menu
    {
      label: 'File',
      submenu: [
        {
          label: 'New Window',
          accelerator: 'CmdOrCtrl+N',
          click: () => { createWindow() }
        },
        { type: 'separator' as const },
        {
          label: 'Open File...',
          accelerator: 'CmdOrCtrl+O',
          click: () => {
            focusedWindow()?.webContents.send('menu-open-file')
          }
        },
        {
          label: 'Save',
          accelerator: 'CmdOrCtrl+S',
          click: () => {
            focusedWindow()?.webContents.send('menu-save-file')
          }
        },
        {
          label: 'Save As...',
          accelerator: 'CmdOrCtrl+Shift+S',
          click: () => {
            focusedWindow()?.webContents.send('menu-save-as')
          }
        },
        {
          label: 'Export',
          submenu: [
            {
              label: 'SRT...',
              click: () => {
                focusedWindow()?.webContents.send('menu-export-srt')
              }
            }
          ]
        },
        { type: 'separator' as const },
        isMac ? { role: 'close' as const } : { role: 'quit' as const }
      ]
    },

    // Edit menu
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' as const },
        { role: 'redo' as const },
        { type: 'separator' as const },
        { role: 'cut' as const },
        { role: 'copy' as const },
        { role: 'paste' as const },
        ...(isMac ? [
          { role: 'pasteAndMatchStyle' as const },
          { role: 'delete' as const },
          { role: 'selectAll' as const },
          { type: 'separator' as const },
          {
            label: 'Speech',
            submenu: [
              { role: 'startSpeaking' as const },
              { role: 'stopSpeaking' as const }
            ]
          }
        ] : [
          { role: 'delete' as const },
          { type: 'separator' as const },
          { role: 'selectAll' as const }
        ])
      ]
    },

    // Speaker menu
    {
      label: 'Speaker',
      submenu: [
        {
          label: 'Rename Speaker...',
          click: () => {
            focusedWindow()?.webContents.send('menu-rename-speaker')
          }
        },
        {
          label: 'Sort Rows by Speaker Similarity',
          click: () => {
            focusedWindow()?.webContents.send('menu-compute-speaker-similarity')
          }
        }
      ]
    },

    // AI Annotations menu
    {
      label: 'AI Annotations',
      submenu: [
        {
          label: 'Caption with Speech Recognizer',
          enabled: false,  // Will be enabled when media is loaded
          id: 'asr-caption',
          click: () => {
            focusedWindow()?.webContents.send('menu-asr-caption')
          }
        },
        {
          label: 'Compute Speaker Embeddings for Segments',
          enabled: false,  // Will be enabled when media is loaded and segments exist
          id: 'asr-embed',
          click: () => {
            focusedWindow()?.webContents.send('menu-asr-embed')
          }
        }
      ]
    },

    // View menu
    {
      label: 'View',
      submenu: [
        { role: 'reload' as const },
        { role: 'forceReload' as const },
        { role: 'toggleDevTools' as const },
        { type: 'separator' as const },
        { role: 'resetZoom' as const },
        { role: 'zoomIn' as const },
        { role: 'zoomOut' as const },
        { type: 'separator' as const },
        { role: 'togglefullscreen' as const }
      ]
    },

    // Window menu
    {
      label: 'Window',
      submenu: [
        { role: 'minimize' as const },
        { role: 'zoom' as const },
        ...(isMac ? [
          { type: 'separator' as const },
          { role: 'front' as const },
          { type: 'separator' as const },
          { role: 'window' as const }
        ] : [
          { role: 'close' as const }
        ])
      ]
    },

    // Help menu
    {
      role: 'help',
      submenu: [
        {
          label: `Caption Editor v${APP_VERSION}`,
          enabled: false
        }
      ]
    }
  ]

  const menu = Menu.buildFromTemplate(template)
  Menu.setApplicationMenu(menu)
}

function createWindow(): BrowserWindow {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    show: process.env.HEADLESS !== 'true',
    // Prevent a bright flash when launching in dark mode.
    backgroundColor: nativeTheme.shouldUseDarkColors ? '#0f1115' : '#ffffff',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,  // Disabled to allow file.path property in drag-and-drop
      webSecurity: true,
      allowRunningInsecureContent: false
    }
  })

  // Set up Content Security Policy
  win.webContents.session.webRequest.onHeadersReceived((details, callback) => {
    const csp = [
      "default-src 'self'",
      "script-src 'self' 'unsafe-inline' 'unsafe-eval'", // unsafe-eval is needed for Vite in dev, unsafe-inline for some libraries
      "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
      "font-src 'self' https://fonts.gstatic.com",
      "img-src 'self' data: blob:",
      "media-src 'self' media: blob:", // Allow our custom media protocol
      "connect-src 'self' ws: http: https:" // Allow dev server connections
    ].join('; ')

    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [csp]
      }
    })
  })

  // In development, load from Vite dev server
  if (process.env.VITE_DEV_SERVER_URL) {
    win.loadURL(process.env.VITE_DEV_SERVER_URL)
    win.webContents.openDevTools()
  } else {
    // In production, load from built files
    win.loadFile(path.join(__dirname, '../dist/index.html'))
  }

  win.on('close', (e) => {
    // In test environment, allow direct closing to avoid hanging E2E tests.
    // In production/dev, we intercept to allow "unsaved changes" confirmation.
    if (!isQuitting && process.env.NODE_ENV !== 'test') {
      e.preventDefault()
      win.webContents.send('app-close')
    }
  })

  // Send any pending file to open once the window is ready
  win.webContents.on('did-finish-load', () => {
    if (fileToOpen) {
      win.webContents.send('open-file', fileToOpen)
      fileToOpen = null
    }
  })

  return win
}

// Track quit state for close-handler
let isQuitting = false

// IPC handler to actually quit/close after confirmation
ipcMain.on('app:quit', (event) => {
  const win = BrowserWindow.fromWebContents(event.sender)
  if (win) {
    if (BrowserWindow.getAllWindows().length <= 1) {
      isQuitting = true
      app.quit()
    } else {
      isQuitting = true
      win.close()
      isQuitting = false
    }
  }
})

/** License acceptance: localStorage is unreliable for packaged `file://` loads; persist under userData. */
const LICENSE_ACCEPTED_FILENAME = 'license-accepted.json'

function licenseAcceptedFilePath(): string {
  return path.join(app.getPath('userData'), LICENSE_ACCEPTED_FILENAME)
}

function readLicenseAcceptedFromDisk(): boolean {
  try {
    const p = licenseAcceptedFilePath()
    if (!existsSync(p)) return false
    const data = JSON.parse(readFileSync(p, 'utf8')) as { accepted?: unknown }
    return data.accepted === true
  } catch {
    return false
  }
}

ipcMain.on('license:getAcceptedSync', (event) => {
  event.returnValue = readLicenseAcceptedFromDisk()
})

ipcMain.handle('license:setAccepted', async () => {
  const p = licenseAcceptedFilePath()
  mkdirSync(path.dirname(p), { recursive: true })
  writeFileSync(p, JSON.stringify({ accepted: true, version: 1 }), 'utf8')
})

if (process.env.NODE_ENV === 'test') {
  ipcMain.handle('license:clearAcceptedForTests', async () => {
    try {
      unlinkSync(licenseAcceptedFilePath())
    } catch {
      // no file
    }
  })
}

// Handle file opening from OS (macOS)
let fileToOpen: string | null = null

app.on('open-file', (event, filePath) => {
  event.preventDefault()

  const toOpen = resolveOpenFilePathPreferSiblingCaptions(filePath)
  const windows = BrowserWindow.getAllWindows()
  if (windows.length > 0) {
    const win = BrowserWindow.getFocusedWindow() || windows[0]
    win.webContents.send('open-file', toOpen)
  } else {
    // Window not ready yet, store for later
    fileToOpen = toOpen
  }
})

app.whenReady().then(() => {
  // Ensure we follow the OS appearance setting (macOS light/dark).
  nativeTheme.themeSource = 'system'

  // Create a custom media protocol handler to securely serve local files.
  // Using a custom protocol is required once webSecurity is enabled.
  protocol.handle('media', async (request) => {
    try {
      const url = new URL(request.url)
      // Extract the path - it should be everything after 'media://local'
      let filePath = decodeURIComponent(url.pathname)

      // Normalize for pathToFileURL (remove leading slash if it's a Windows drive letter)
      if (process.platform === 'win32' && filePath.startsWith('/')) {
        filePath = filePath.substring(1)
      }

      // Detect MIME type based on extension
      const ext = path.extname(filePath).toLowerCase()
      const contentType = MIME_TYPES[ext] || 'application/octet-stream'

      const stats = await fs.stat(filePath)
      const fileSize = stats.size

      // Parse Range header for byte-range serving.
      // We handle ranges manually instead of relying on net.fetch because
      // Electron's net.fetch on file:// URLs throws ERR_REQUEST_RANGE_NOT_SATISFIABLE
      // for requests near EOF, causing Chromium to retry in an infinite loop.
      const rangeHeader = request.headers.get('Range')

      if (rangeHeader) {
        const match = rangeHeader.match(/bytes=(\d+)-(\d*)/)
        if (match) {
          const start = parseInt(match[1], 10)
          const end = match[2] ? parseInt(match[2], 10) : fileSize - 1

          // Validate range
          if (start >= fileSize) {
            const headers = new Headers()
            headers.set('Content-Range', `bytes */${fileSize}`)
            headers.set('Accept-Ranges', 'bytes')
            return new Response(null, { status: 416, statusText: 'Range Not Satisfiable', headers })
          }

          const clampedEnd = Math.min(end, fileSize - 1)
          const contentLength = clampedEnd - start + 1

          const { createReadStream } = await import('fs')
          const stream = createReadStream(filePath, { start, end: clampedEnd })
          const readable = new ReadableStream({
            start(controller) {
              stream.on('data', (chunk: Buffer) => controller.enqueue(chunk))
              stream.on('end', () => controller.close())
              stream.on('error', (err) => controller.error(err))
            },
            cancel() { stream.destroy() }
          })

          const headers = new Headers()
          headers.set('Content-Type', contentType)
          headers.set('Content-Length', contentLength.toString())
          headers.set('Content-Range', `bytes ${start}-${clampedEnd}/${fileSize}`)
          headers.set('Accept-Ranges', 'bytes')

          return new Response(readable, { status: 206, statusText: 'Partial Content', headers })
        }
      }

      // Non-range request: serve the full file
      const response = await net.fetch(pathToFileURL(filePath).toString(), {
        bypassCustomProtocolHandlers: true,
        method: request.method,
      })

      const headers = new Headers(response.headers)
      headers.set('Content-Type', contentType)
      headers.set('Accept-Ranges', 'bytes')
      if (!headers.has('Content-Length')) {
        headers.set('Content-Length', fileSize.toString())
      }

      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers
      })
    } catch (error) {
      console.error('[main] media:// protocol error:', error)
      return new Response('Invalid media URL', { status: 400 })
    }
  })

  createMenu()
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })

  // Check if app was launched with a file path (Windows/Linux/macOS)
  // On macOS, this handles test scenarios where files are passed as arguments
  // In production, macOS uses the 'open-file' event instead
  if (process.argv.length >= 2) {
    const filePath = process.argv[process.argv.length - 1]
    const lower = (filePath || '').toLowerCase()
    const ext = path.extname(lower)
    if (filePath && !filePath.startsWith('-') && (lower.endsWith(CAPTIONS_JSON_SUFFIX) || lower.endsWith('.captions_json') || lower.endsWith('.srt') || ext in MIME_TYPES)) {
      fileToOpen = resolveOpenFilePathPreferSiblingCaptions(filePath)
    }
  }

  // Handle files dropped from preload — relay back to the sender window
  ipcMain.on('files-dropped', (event, filePaths: string[]) => {
    const win = BrowserWindow.fromWebContents(event.sender)
    if (win) {
      win.webContents.send('files-dropped', filePaths)
    }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

// File operations with proper permission handling

/**
 * Open file picker dialog and return file info
 */
ipcMain.handle('dialog:openFile', async (event, options?: {
  filters?: Array<{ name: string; extensions: string[] }>,
  properties?: Array<'openFile' | 'multiSelections'>
}) => {
  const win = windowForEvent(event)
  if (!win) return null

  const result = await dialog.showOpenDialog(win, {
    properties: options?.properties || ['openFile'],
    filters: options?.filters || [
      { name: 'All Supported Files', extensions: all_files },
      { name: 'Captions Files (*.captions_json5)', extensions: captions_json5_files },
      { name: 'SRT Files', extensions: srt_files },
      { name: 'Media Files', extensions: media_files }
    ]
  })

  if (result.canceled || result.filePaths.length === 0) {
    return null
  }

  return result.filePaths
})

/**
 * Read file contents - handles security-scoped bookmarks on macOS
 */
ipcMain.handle('file:read', async (_event, filePath: string) => {
  try {
    // On macOS, start accessing the security-scoped resource
    if (process.platform === 'darwin' && fileBookmarks.has(filePath)) {
      const _bookmark = fileBookmarks.get(filePath)!
      // In a real implementation, you'd use app.startAccessingSecurityScopedResource
      // For now, we rely on the dialog.showOpenDialog providing temporary access
    }

    const content = await fs.readFile(filePath, 'utf-8')
    return { success: true, content, filePath }
  } catch (error) {
    console.error('Error reading file:', error)
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error reading file'
    }
  }
})

/**
 * Write file contents - prompts for save location
 */
ipcMain.handle('file:save', async (event, options: {
  content: string,
  defaultPath?: string,
  suggestedName?: string
}) => {
  const win = windowForEvent(event)
  if (!win) return { success: false, error: 'No window available' }

  try {
    const result = await dialog.showSaveDialog(win, {
      defaultPath: options.suggestedName || `captions${CAPTIONS_JSON_SUFFIX}`,
      filters: [
        { name: 'Captions Files (*.captions_json5)', extensions: captions_json5_files },
        { name: 'All Files', extensions: ['*'] }
      ]
    })

    if (result.canceled || !result.filePath) {
      return { success: false, error: 'Save canceled' }
    }

    let targetPath = result.filePath
    if (!targetPath.toLowerCase().endsWith(CAPTIONS_JSON_SUFFIX)) {
      targetPath = targetPath + CAPTIONS_JSON_SUFFIX
    }

    await fs.writeFile(targetPath, options.content, 'utf-8')

    return { success: true, filePath: targetPath }
  } catch (error) {
    console.error('Error saving file:', error)
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error saving file'
    }
  }
})

/**
 * Write SRT file contents - prompts for save location
 */
ipcMain.handle('file:saveSrt', async (event, options: {
  content: string,
  suggestedName?: string
}) => {
  const win = windowForEvent(event)
  if (!win) return { success: false, error: 'No window available' }

  try {
    const result = await dialog.showSaveDialog(win, {
      defaultPath: options.suggestedName || 'captions.srt',
      filters: [
        { name: 'SRT Files', extensions: srt_files },
        { name: 'All Files', extensions: ['*'] }
      ]
    })

    if (result.canceled || !result.filePath) {
      return { success: false, error: 'Save canceled' }
    }

    let targetPath = result.filePath
    if (!targetPath.toLowerCase().endsWith('.srt')) {
      targetPath = targetPath + '.srt'
    }

    await fs.writeFile(targetPath, options.content, 'utf-8')

    return { success: true, filePath: targetPath }
  } catch (error) {
    console.error('Error saving SRT file:', error)
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error saving file'
    }
  }
})

/**
 * Save to existing file (already has permission from previous open/save)
 */
ipcMain.handle('file:saveExisting', async (_event, options: {
  filePath: string,
  content: string
}) => {
  try {
    // On macOS, start accessing the security-scoped resource
    if (process.platform === 'darwin' && fileBookmarks.has(options.filePath)) {
      const _bookmark = fileBookmarks.get(options.filePath)!
      // In a real implementation, you'd use app.startAccessingSecurityScopedResource
    }

    await fs.writeFile(options.filePath, options.content, 'utf-8')

    return { success: true, filePath: options.filePath }
  } catch (error) {
    console.error('Error saving file:', error)
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error saving file'
    }
  }
})

/**
 * Get file stats
 */
ipcMain.handle('file:stat', async (_event, filePath: string) => {
  try {
    const stats = await fs.stat(filePath)
    return {
      success: true,
      exists: true,
      isFile: stats.isFile(),
      isDirectory: stats.isDirectory(),
      size: stats.size,
      modified: stats.mtime.toISOString()
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return { success: true, exists: false }
    }
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error checking file'
    }
  }
})

/**
 * Show file in Finder/Explorer
 */
ipcMain.handle('file:showInFolder', async (_event, filePath: string) => {
  try {
    shell.showItemInFolder(filePath)
    return { success: true }
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }
  }
})

/**
 * Convert file path to protocol URL for media loading
 */
ipcMain.handle('file:toURL', async (_event, filePath: string) => {
  try {
    // Ensure the file exists and we can access it
    await fs.access(filePath)

    // Return media:// URL instead of file://
    // Using media://local/path format for clean URL parsing
    const url = `media://local${filePath}`
    return { success: true, url, filePath }
  } catch (error) {
    console.error('Error converting file to URL:', error)
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Cannot access file'
    }
  }
})

/**
 * Update menu item enabled state
 */
ipcMain.on('menu:updateAsrEnabled', (_event, options: boolean | { caption?: boolean; embed?: boolean }) => {
  const menu = Menu.getApplicationMenu()
  if (menu) {
    if (typeof options === 'boolean') {
      const asrItem = menu.getMenuItemById('asr-caption')
      if (asrItem) {
        asrItem.enabled = options
      }
    } else {
      if (options.caption !== undefined) {
        const asrItem = menu.getMenuItemById('asr-caption')
        if (asrItem) asrItem.enabled = options.caption
      }
      if (options.embed !== undefined) {
        const embedItem = menu.getMenuItemById('asr-embed')
        if (embedItem) embedItem.enabled = options.embed
      }
    }
  }
})

/**
 * Ensures the Rust ASR binaries (transcribe-rs, embed-rs) for the current
 * APP_VERSION are present in ~/.cache/caption_editor/bin/rust-asr-v${APP_VERSION}/.
 *
 * Downloads them from the matching GitHub Release on first use:
 *   https://github.com/<repo>/releases/download/v${APP_VERSION}/<name>-v${APP_VERSION}-darwin-arm64
 *
 * Cache key is the version directory itself — so an APP_VERSION bump
 * invalidates the cache automatically and forces a re-download, while
 * downgrading to a previous version keeps that version's binaries
 * around for offline use.
 */
async function ensureRustAsrBinaries(onLog?: (msg: string) => void): Promise<{
  transcribeRs: string
  embedRs: string
}> {
  const cacheDir = path.join(os.homedir(), '.cache', 'caption_editor')
  const versionDir = path.join(cacheDir, 'bin', `rust-asr-v${APP_VERSION}`)
  if (!existsSync(versionDir)) {
    mkdirSync(versionDir, { recursive: true })
  }

  const transcribeRsPath = path.join(versionDir, 'transcribe-rs')
  const embedRsPath = path.join(versionDir, 'embed-rs')

  if (existsSync(transcribeRsPath) && existsSync(embedRsPath)) {
    return { transcribeRs: transcribeRsPath, embedRs: embedRsPath }
  }

  const log = (msg: string) => {
    console.log(`[main] ${msg}`)
    if (onLog) onLog(msg + '\n')
  }

  if (process.arch !== 'arm64' || process.platform !== 'darwin') {
    throw new Error(
      `Rust ASR binaries are only published for darwin-arm64 (current: ${process.platform}-${process.arch}). ` +
      `Build locally with bazelisk and point CAPTION_EDITOR_TRANSCRIBE_RS_BIN at the binary.`
    )
  }

  // The repo's GitHub Release page hosts assets at predictable URLs. Strip
  // the `git+`/`.git` prefixes from ASR_GITHUB_REPO that exist for the uvx
  // pip-spec format; the bare https://github.com/<owner>/<repo> form is
  // what `releases/download/...` paths are built from.
  const httpRepo = ASR_GITHUB_REPO
    .replace(/^git\+/, '')
    .replace(/\.git$/, '')
  const tag = `v${APP_VERSION}`
  const suffix = `-${tag}-darwin-arm64`

  log(`Rust ASR binaries missing for ${tag}. Downloading from GitHub Release...`)

  const downloadOne = async (name: 'transcribe-rs' | 'embed-rs', destPath: string) => {
    const assetName = `${name}${suffix}`
    const url = `${httpRepo}/releases/download/${tag}/${assetName}`
    log(`  ${name}: ${url}`)
    const response = await net.fetch(url, { redirect: 'follow' })
    if (!response.ok) {
      throw new Error(
        `Failed to download ${assetName} from ${url}: HTTP ${response.status} ${response.statusText}. ` +
        `Has the v${APP_VERSION} release been published with the Rust binaries attached?`
      )
    }
    const buf = Buffer.from(await response.arrayBuffer())
    await fs.writeFile(destPath, buf)
    await fs.chmod(destPath, 0o755)
  }

  try {
    await downloadOne('transcribe-rs', transcribeRsPath)
    await downloadOne('embed-rs', embedRsPath)
    log(`Rust ASR binaries installed in ${versionDir}`)
    return { transcribeRs: transcribeRsPath, embedRs: embedRsPath }
  } catch (err) {
    // Don't leave half-downloaded binaries that would short-circuit the
    // existsSync check on the next call.
    for (const p of [transcribeRsPath, embedRsPath]) {
      if (existsSync(p)) {
        try { await fs.unlink(p) } catch { /* ignore */ }
      }
    }
    throw err
  }
}

/**
 * Common helper to run ASR tools (transcribe, embed, etc.)
 */
interface AsrResult {
  success: boolean
  script?: string
  processId?: string
  error?: string
  canceled?: boolean
}

interface ActiveProcess {
  proc: ChildProcess
  cancel: () => void
}

async function runAsrTool(options: {
  script: 'transcribe_cli.py' | 'embed_cli.py',
  inputPath: string,
  model?: string,
  chunkSize?: number,
  remuxMp3?: boolean,
  senderWebContents?: Electron.WebContents
}): Promise<AsrResult> {
  const { script, inputPath, model, chunkSize, remuxMp3 } = options

  // Store process for cancellation
  const processId = Date.now().toString()

  const shouldMirrorAsrOutputToMainStdio =
    process.env.CAPTION_EDITOR_MIRROR_ASR_OUTPUT_TO_STDIO === '1' ||
    process.env.NODE_ENV === 'test'

  const appendTail = (current: string, addition: string, maxChars: number) => {
    const combined = current + addition
    if (combined.length <= maxChars) return combined
    return combined.slice(combined.length - maxChars)
  }

  let stdoutTail = ''
  let stderrTail = ''

  const sendOutput = (type: 'stdout' | 'stderr', data: string) => {
    options.senderWebContents?.send('asr:output', { processId, type, data })

    if (shouldMirrorAsrOutputToMainStdio) {
      const prefix = `[asr:${processId}:${type}] `
      if (type === 'stdout') {
        process.stdout.write(prefix + data)
      } else {
        process.stderr.write(prefix + data)
      }
    }
  }

  // Determine if we're in dev mode
  const runFromCodeTree = process.env.CAPTION_EDITOR_RUN_TRANSCRIBE_FROM_CODE_TREE === '1'
  const isDev = runFromCodeTree || process.env.NODE_ENV === 'development' || process.env.VITE_DEV_SERVER_URL

  let pythonCommand: string
  let pythonArgs: string[]
  let cwd: string

  // Rust ASR bypass. When CAPTION_EDITOR_TRANSCRIBE_RS_BIN or _EMBED_RS_BIN are
  // set, invoke the //transcribe_rs/ Rust binary directly instead of uvx/Python.
  // Args are wire-compatible with the Python CLI (we kept --model, --chunk-size,
  // --remux-mp3 flag names identical so this swap is a drop-in). Setting just
  // one of the two vars is fine — e.g. test transcribe-rs while still embedding
  // via the Python pipeline.
  const transcribeRsBin = process.env.CAPTION_EDITOR_TRANSCRIBE_RS_BIN
  const embedRsBin = process.env.CAPTION_EDITOR_EMBED_RS_BIN
  const useRust =
    (script === 'transcribe_cli.py' && transcribeRsBin) ||
    (script === 'embed_cli.py' && embedRsBin)

  if (useRust) {
    pythonCommand = (script === 'transcribe_cli.py' ? transcribeRsBin : embedRsBin)!
    if (!existsSync(pythonCommand)) {
      throw new Error(`Rust ASR binary not found at ${pythonCommand}`)
    }
    pythonArgs = [inputPath]
    if (script === 'transcribe_cli.py') {
      if (chunkSize !== undefined) pythonArgs.push('--chunk-size', chunkSize.toString())
      if (model) pythonArgs.push('--model', model)
      if (remuxMp3) pythonArgs.push('--remux-mp3')
      // The Python --embed default is true, so the GUI's "transcribe" handler
      // expects embeddings to be present afterward. transcribe-rs defaults
      // to --embed too; it shells out to a sibling embed-rs binary. Help
      // that resolution along by passing --embed-bin if we know it.
      if (embedRsBin && existsSync(embedRsBin)) {
        pythonArgs.push('--embed-bin', embedRsBin)
      }
    } else {
      // embed_cli.py → embed-rs
      if (model) pythonArgs.push('--model', model)
    }
    cwd = os.tmpdir()
  } else if (isDev) {
    const codeTreeRoot = process.env.CAPTION_EDITOR_RUN_TRANSCRIBE_FROM_CODE_TREE === '1'
      ? (process.env.CAPTION_EDITOR_CODE_TREE_ROOT || path.join(__dirname, '..'))
      : path.join(__dirname, '..')

    pythonCommand = 'uv'
    pythonArgs = ['run', 'python', script, inputPath]
    cwd = path.join(codeTreeRoot, 'transcribe')

    if (script === 'transcribe_cli.py' && chunkSize !== undefined) {
      pythonArgs.push('--chunk-size', chunkSize.toString())
    }
    if (model) pythonArgs.push('--model', model)
    if (script === 'transcribe_cli.py' && remuxMp3) pythonArgs.push('--remux-mp3')

    const scriptPath = path.join(cwd, script)
    if (!existsSync(scriptPath)) {
      throw new Error(`${script} not found at ${scriptPath}`)
    }
  } else {
    // Production mode: download the Rust ASR binaries for the current
    // APP_VERSION from the GitHub Release and invoke them directly.
    // First call pays the network cost (~80 MB across both binaries),
    // subsequent calls hit the on-disk cache in
    // ~/.cache/caption_editor/bin/rust-asr-v${APP_VERSION}/.
    const { transcribeRs, embedRs } = await ensureRustAsrBinaries((msg) =>
      sendOutput('stdout', msg),
    )
    pythonCommand = (script === 'transcribe_cli.py' ? transcribeRs : embedRs)
    pythonArgs = [inputPath]
    if (script === 'transcribe_cli.py') {
      if (chunkSize !== undefined) pythonArgs.push('--chunk-size', chunkSize.toString())
      if (model) pythonArgs.push('--model', model)
      if (remuxMp3) pythonArgs.push('--remux-mp3')
      // transcribe-rs auto-embeds via a sibling embed-rs binary; the
      // download lands both binaries in the same dir so its
      // auto-discovery (current_exe parent → embed-rs) already works,
      // but passing --embed-bin makes it explicit and survives any
      // future install layout changes.
      pythonArgs.push('--embed-bin', embedRs)
    } else {
      // embed_cli.py → embed-rs
      if (model) pythonArgs.push('--model', model)
    }
    cwd = os.tmpdir()
  }

  options.senderWebContents?.send('asr:started', { processId })

  const { spawn } = await import('child_process')
  const binDir = path.join(os.homedir(), '.cache', 'caption_editor', 'bin')
  const env = { ...process.env, PATH: `${binDir}${path.delimiter}${process.env.PATH || ''}` }

  // Use a temporary directory for CWD to avoid issues with spaces in project paths 
  // (some tools like uvx might have issues with spaces in current directory)
  if (!cwd) {
    cwd = os.tmpdir()
  }

  let canceled = false

  return new Promise((resolve, reject) => {
    // Start process in its own process group so we can kill its children
    const proc = spawn(pythonCommand, pythonArgs, {
      cwd,
      env,
      detached: process.platform !== 'win32'
    })

    activeProcesses.set(processId, {
      proc,
      cancel: () => {
        canceled = true
        if (process.platform === 'win32') {
          proc.kill()
        } else {
          try {
            // Kill the entire process group
            process.kill(-proc.pid!, 'SIGTERM')
          } catch {
            // Fallback if PGID killing fails
            proc.kill('SIGTERM')
          }
        }
      }
    })

    proc.stdout?.on('data', (data) => {
      const chunk = data.toString()
      stdoutTail = appendTail(stdoutTail, chunk, 20_000)
      sendOutput('stdout', chunk)
    })
    proc.stderr?.on('data', (data) => {
      const chunk = data.toString()
      stderrTail = appendTail(stderrTail, chunk, 20_000)
      sendOutput('stderr', chunk)
    })

    proc.on('close', (code) => {
      activeProcesses.delete(processId)
      if (code === 0) {
        resolve({ success: true, script, processId })
      } else if (canceled || code === 143) {
        console.log(`[main] ASR ${script} process ${processId} canceled or terminated with code ${code}`)
        resolve({ success: false, error: 'Canceled', canceled: true })
      } else {
        const errorMsg = `Process exited with code ${code}`
        console.error(`[main] ASR ${script} ${errorMsg}`)
        const tail = (stderrTail || stdoutTail).trim()
        reject(new Error(tail ? `${errorMsg}\n\n--- process output (tail) ---\n${tail}` : errorMsg))
      }
    })

    proc.on('error', (err) => {
      activeProcesses.delete(processId)
      if (canceled) {
        resolve({ success: false, error: 'Canceled', canceled: true })
      } else {
        reject(err)
      }
    })
  })
}

/**
 * Run ASR transcription on media file
 */
ipcMain.handle('asr:transcribe', async (event, options: {
  mediaFilePath: string,
  model?: string,
  chunkSize?: number,
  remuxMp3?: boolean
}) => {
  // If the output captions file already exists, back it up instead of letting the CLI fail
  const captionsPath = options.mediaFilePath.replace(path.extname(options.mediaFilePath), CAPTIONS_JSON_SUFFIX)
  try {
    await fs.access(captionsPath)
    // File exists — find a backup name that doesn't collide
    const backupPath = await findBackupPath(captionsPath, async (p) => {
      try { await fs.access(p); return true } catch { return false }
    })
    console.log(`[main] Backing up existing captions file: ${captionsPath} -> ${backupPath}`)
    await fs.rename(captionsPath, backupPath)
  } catch {
    // File doesn't exist, nothing to back up
  }

  const result = await runAsrTool({
    script: 'transcribe_cli.py',
    inputPath: options.mediaFilePath,
    model: options.model,
    chunkSize: options.chunkSize,
    remuxMp3: options.remuxMp3,
    senderWebContents: event.sender
  })

  if (result.success) {
    try {
      const hr = '='.repeat(76)
      console.log(hr)
      console.log('[main] asr:transcribe — Python finished OK. Reading captions file from disk (large files can take a moment)…')
      console.log('[main] asr:transcribe — path:', captionsPath)
      const readStart = Date.now()
      const content = await fs.readFile(captionsPath, 'utf-8')
      const readMs = Date.now() - readStart
      console.log(`[main] asr:transcribe — read ${content.length} characters in ${readMs}ms; sending to renderer over IPC (also can take a moment)`)
      console.log(hr)
      return {
        ...result,
        captionsPath,
        content
      }
    } catch (err) {
      console.error('[main] Failed to read generated captions JSON file:', err)
      return {
        success: false,
        error: `Transcription succeeded but failed to read result file: ${err instanceof Error ? err.message : 'Unknown error'}`
      }
    }
  }

  return result
})

/**
 * Run speaker embedding on captions JSON file
 */
ipcMain.handle('asr:embed', async (event, options: {
  captionsPath: string,
  model?: string
}) => {
  const result = await runAsrTool({
    script: 'embed_cli.py',
    inputPath: options.captionsPath,
    model: options.model,
    senderWebContents: event.sender
  })

  if (result.success) {
    try {
      const hr = '='.repeat(76)
      console.log(hr)
      console.log('[main] asr:embed — Python finished OK. Reading captions file from disk (embeddings make files large)…')
      console.log('[main] asr:embed — path:', options.captionsPath)
      const readStart = Date.now()
      const content = await fs.readFile(options.captionsPath, 'utf-8')
      const readMs = Date.now() - readStart
      console.log(`[main] asr:embed — read ${content.length} characters in ${readMs}ms; sending to renderer over IPC`)
      console.log(hr)
      return {
        ...result,
        content
      }
    } catch (err) {
      console.error('[main] Failed to read captions JSON file after embedding:', err)
      return {
        success: false,
        error: `Embedding succeeded but failed to read result file: ${err instanceof Error ? err.message : 'Unknown error'}`
      }
    }
  }

  return result
})



/**
 * Cancel running ASR process
 */
ipcMain.handle('asr:cancel', async (_event, processId: string) => {
  const item = activeProcesses.get(processId)
  if (item) {
    console.log(`[main] Cancelling ASR process ${processId}`)
    item.cancel()
    activeProcesses.delete(processId)
    return { success: true }
  }
  return { success: false, error: 'Process not found' }
})

// Store active ASR processes
const activeProcesses = new Map<string, ActiveProcess>()

// Handle file drops from system
ipcMain.handle('file:processDroppedFiles', async (_event, filePaths: string[]) => {
  const t0 = performance.now()
  console.log('[main] processDroppedFiles called for', filePaths.length, 'files')

  type DroppedFileResult =
    | { type: 'captions_json5'; filePath: string; fileName: string; content: string }
    | { type: 'srt'; filePath: string; fileName: string; content: string }
    | { type: 'media'; filePath: string; fileName: string; url: string }

  const results: DroppedFileResult[] = []

  for (const filePath of filePaths) {
    try {
      const stats = await fs.stat(filePath)
      if (!stats.isFile()) continue

      const ext = path.extname(filePath).toLowerCase()
      const extensionWithoutDot = ext.substring(1)
      const lowerPath = filePath.toLowerCase()

      if (lowerPath.endsWith(CAPTIONS_JSON_SUFFIX) || lowerPath.endsWith('.captions_json')) {
        const content = await fs.readFile(filePath, 'utf-8')
        results.push({
          type: 'captions_json5',
          filePath,
          fileName: path.basename(filePath),
          content
        })
        console.log(`[main] Loaded captions JSON: ${filePath}`)
      } else if (srt_files.includes(extensionWithoutDot)) {
        const content = await fs.readFile(filePath, 'utf-8')
        results.push({
          type: 'srt',
          filePath,
          fileName: path.basename(filePath),
          content
        })
        console.log(`[main] Loaded SRT: ${filePath}`)
      } else if (media_files.includes(extensionWithoutDot)) {
        const url = `media://local${filePath}`
        results.push({
          type: 'media',
          filePath,
          fileName: path.basename(filePath),
          url
        })
        console.log(`[main] Created media URL for: ${filePath}`)
      } else {
        console.log(`[main] Skipping unsupported file type: ${ext}`)
      }
    } catch (error) {
      console.error(`[main] Error processing file ${filePath}:`, error)
    }
  }

  console.log(`[main] processDroppedFiles done in ${(performance.now() - t0).toFixed(1)} ms`)
  return results
})
