import { AudioLines, Compass, Library, type LucideIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { NavLink } from "react-router";
import { health } from "../lib/api";
import { Logo } from "./Logo";

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  soon?: boolean;
}

const NAV: NavItem[] = [
  { to: "/", label: "Identify", icon: AudioLines },
  { to: "/library", label: "Library", icon: Library },
  { to: "/discover", label: "Discover", icon: Compass },
];

/** Desktop left sidebar. */
export function Sidebar() {
  const [songs, setSongs] = useState<number | null>(null);
  useEffect(() => {
    health().then((h) => setSongs(h.songs)).catch(() => {});
  }, []);

  return (
    <aside className="hidden w-60 flex-col gap-6 border-r border-line/60 bg-surface p-4 md:flex">
      <div className="px-2 pt-2">
        <Logo />
      </div>
      <nav aria-label="Main" className="flex flex-col gap-1">
        {NAV.map((item) =>
          item.soon ? (
            <span key={item.to} className="flex cursor-not-allowed items-center gap-3 rounded-lg px-3 py-2.5 font-semibold text-subtle" aria-disabled="true">
              <item.icon className="size-5" aria-hidden="true" />
              {item.label}
              <span className="ml-auto rounded-full border border-line px-2 py-0.5 text-[10px] font-bold tracking-wider uppercase">Soon</span>
            </span>
          ) : (
            <NavLink
              key={item.to}
              to={item.to}
              end
              className={({ isActive }) =>
                `relative flex items-center gap-3 rounded-lg px-3 py-2.5 font-semibold transition-colors duration-150 ${
                  isActive
                    ? "bg-elevated text-fg before:absolute before:inset-y-2 before:left-0 before:w-1 before:rounded-full before:bg-accent"
                    : "text-muted hover:text-fg"
                }`
              }
            >
              <item.icon className="size-5" aria-hidden="true" />
              {item.label}
            </NavLink>
          ),
        )}
      </nav>
      <div className="mt-auto rounded-lg bg-elevated p-3 text-xs text-muted">
        <p className="font-semibold text-fg">{songs === null ? "Library" : `${songs.toLocaleString()} tracks`}</p>
        <p className="mt-1">Creative Commons music from the Free Music Archive.</p>
      </div>
    </aside>
  );
}

/** Mobile bottom tab bar (replaces the sidebar below the md breakpoint). */
export function MobileTabBar() {
  return (
    <nav aria-label="Main" className="grid grid-cols-3 border-t border-line/60 bg-surface pb-[env(safe-area-inset-bottom)] md:hidden">
      {NAV.map((item) =>
        item.soon ? (
          <span key={item.to} className="flex flex-col items-center gap-1 py-2 text-[11px] font-semibold text-subtle" aria-disabled="true">
            <item.icon className="size-5" aria-hidden="true" />
            {item.label}
          </span>
        ) : (
          <NavLink
            key={item.to}
            to={item.to}
            end
            className={({ isActive }) =>
              `flex flex-col items-center gap-1 py-2 text-[11px] font-semibold transition-colors ${isActive ? "text-fg" : "text-muted"}`
            }
          >
            {({ isActive }) => (
              <>
                <item.icon className={`size-5 ${isActive ? "text-accent" : ""}`} aria-hidden="true" />
                {item.label}
              </>
            )}
          </NavLink>
        ),
      )}
    </nav>
  );
}
