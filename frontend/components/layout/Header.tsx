"use client";

import React, { useEffect, useState } from "react";
import PillNav from "@/components/ui/PillNav";
import PillButton from "@/components/ui/PillButton";
import Link from "next/link";
import { createClient } from "@/utils/supabase/client";
import { useRouter } from "next/navigation";

export function Header() {
  const [user, setUser] = useState<import("@supabase/supabase-js").User | null>(null);
  const router = useRouter();
  const supabase = createClient();

  useEffect(() => {
    const getUser = async () => {
      const {
        data: { user },
      } = await supabase.auth.getUser();
      setUser(user);
    };
    getUser();

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });

    return () => {
      subscription.unsubscribe();
    };
  }, []);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    router.refresh();
  };

  return (
    <header className="fixed top-0 w-full z-50 pointer-events-none">
      {/* Left Side Logo - Absolute Positioned */}
      <div className="absolute top-4 left-6 pointer-events-auto z-50">
        <Link href="/" className="flex items-center gap-2">
          <span className="text-2xl font-extrabold tracking-tight text-white/90">
            Sign<span className="gradient-text">Bridge</span>
          </span>
        </Link>
      </div>

      {/* Pill Navbar - Centered */}
      <div className="pointer-events-auto">
        <PillNav
          items={[
            { label: "Home", href: "/" },
            { label: "Text to Sign", href: "/text-to-sign" },
            { label: "Sign to Text", href: "/sign-to-text" },
          ]}
          baseColor="#000000"
          pillColor="#ffffff"
          hoveredPillTextColor="#000000"
          pillTextColor="#a1a1aa"
        />
      </div>

      {/* Right Side Actions - Absolute Positioned to match layout */}
      <div className="absolute top-4 right-6 pointer-events-auto hidden md:block">
        {user ? (
          <div className="flex items-center gap-3">
            {user.user_metadata?.avatar_url && (
              <img
                src={user.user_metadata.avatar_url}
                alt={user.user_metadata?.full_name || "Profile"}
                className="w-10 h-10 rounded-full border-2 border-white/10 shadow-xl object-cover hover:scale-105 transition-transform duration-300 pointer-events-auto"
                referrerPolicy="no-referrer"
              />
            )}
            <PillButton
              label="Sign Out"
              onClick={handleSignOut}
              className="font-semibold shadow-lg text-sm"
              baseColor="#18181b"
              hoverColor="#dc2626"
              textColor="#e4e4e7"
              hoverTextColor="#ffffff"
            />
          </div>
        ) : (
          <PillButton
            href="/login"
            label="Log in"
            className="font-semibold shadow-lg text-sm"
            baseColor="linear-gradient(135deg, #6366f1 0%, #a855f7 100%)"
            hoverColor="#ffffff"
            textColor="#ffffff"
            hoverTextColor="#000000"
          />
        )}
      </div>
    </header>
  );
}
