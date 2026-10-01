import { BrowserRouter, Route, Routes } from "react-router";
import { MobileTabBar, Sidebar } from "./components/Navigation";
import { PlayerBar } from "./components/PlayerBar";
import { DiscoverPage } from "./pages/DiscoverPage";
import { IdentifyPage } from "./pages/IdentifyPage";
import { LibraryPage } from "./pages/LibraryPage";
import { PlayerProvider } from "./player/PlayerContext";

/**
 * App shell: sidebar | scrolling main content, with the player bar (and on
 * mobile the tab bar) pinned below. Only <main> scrolls, so the bars never
 * move and the player keeps playing across page navigation.
 */
export function App() {
  return (
    <BrowserRouter>
      <PlayerProvider>
        <div className="flex h-dvh flex-col">
          <div className="flex min-h-0 flex-1">
            <Sidebar />
            <main className="min-w-0 flex-1 overflow-y-auto bg-gradient-to-b from-elevated/60 to-bg to-[28rem]">
              <Routes>
                <Route path="/" element={<IdentifyPage />} />
                <Route path="/library" element={<LibraryPage />} />
                <Route path="/discover" element={<DiscoverPage />} />
                <Route path="*" element={<IdentifyPage />} />
              </Routes>
            </main>
          </div>
          <PlayerBar />
          <MobileTabBar />
        </div>
      </PlayerProvider>
    </BrowserRouter>
  );
}
